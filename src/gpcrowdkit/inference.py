"""Training utilities.

Minibatching
------------
Only *item indices* are shuffled. Features, annotations and ``q(Z)`` rows are
all gathered from the same index vector downstream, so they cannot fall out of
alignment. The reference implementation instead constructs three separate
``Minibatch`` objects -- over ``X``, over the label array, and over the index
array -- kept in step only by passing each the same ``seed=0``. Nothing
enforces that agreement, and if it ever fails the model simply pairs features
with the wrong annotations and trains to a worse optimum, silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import tensorflow as tf

from .data import CrowdLabels
from .models import GPCrowdModel

__all__ = ["batch_iterator", "train", "TrainingHistory"]


class _EarlyStopper:
    """Stops training once a moving average of the ELBO stops improving.

    A single minibatch's ELBO is a noisy estimate of the full-dataset ELBO --
    which item indices happened to be sampled matters, especially at small
    ``batch_size``. Comparing consecutive raw values would trigger on that
    noise rather than on genuine convergence, so this smooths over
    ``window`` iterations first and tracks the best smoothed value seen so
    far, the standard "smoothed moving average with patience" early-stopping
    recipe.

    This is what lets `train` give each `AnnotatorModel` strategy as many
    iterations as *it* needs rather than a number picked for the whole
    comparison: a low-capacity strategy's ELBO plateaus quickly and this
    stops soon after, while a high-capacity one keeps finding improvements
    and keeps training.
    """

    def __init__(self, patience: int, min_delta: float, window: int) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._window: list[float] = []
        self._window_size = window
        # None, not -inf: -inf + min_delta * abs(-inf) is nan (inf - inf) in
        # IEEE-754, which would make the very first comparison silently False
        # forever and never let self._best leave -inf -- an off-by-a-NaN bug
        # that made every strategy "stop improving" on its first possible
        # check, regardless of whether it actually was.
        self._best: float | None = None
        self._since_improved = 0

    def should_stop(self, elbo_value: float) -> bool:
        """Records one iteration's ELBO and reports whether training should stop now."""
        self._window.append(elbo_value)
        if len(self._window) > self._window_size:
            self._window.pop(0)
        if len(self._window) < self._window_size:
            return False  # not enough history yet for a stable moving average

        smoothed = sum(self._window) / len(self._window)
        if self._best is None:
            self._best = smoothed  # first full window: nothing to compare against yet
            return False

        # Relative threshold: an absolute one would need re-tuning per dataset,
        # since ELBO magnitude scales with N, L and the number of classes.
        threshold = self._best + self._min_delta * (abs(self._best) + 1e-8)
        if smoothed > threshold:
            self._best = smoothed
            self._since_improved = 0
        else:
            self._since_improved += 1
        return self._since_improved >= self._patience


@dataclass
class TrainingHistory:
    """Per-iteration trace of the ELBO and each of its components.

    Storing the decomposition rather than the total alone is what makes a bad
    run diagnosable. The characteristic crowdsourcing failure is a rising ELBO
    in which ``crowd`` climbs steadily while ``latent`` stays flat: the model is
    fitting the annotators and ignoring the features, so it will predict
    nothing useful on unannotated data.

    Attributes:
        elbo (list[float]): Total scaled ELBO per iteration.
        latent (list[float]): Latent GP evidence term.
        crowd (list[float]): Crowdsourcing evidence term.
        entropy (list[float]): Entropy of ``q(Z)``.
        kl_latent (list[float]): ``KL(q(u) || p(u))``.
        kl_annotator (list[float]): ``KL(q(R) || p(R))``.
    """

    elbo: list[float] = field(default_factory=list)
    latent: list[float] = field(default_factory=list)
    crowd: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    kl_latent: list[float] = field(default_factory=list)
    kl_annotator: list[float] = field(default_factory=list)


def batch_iterator(num_items: int, batch_size: int | None, seed: int = 0) -> Iterator[tf.Tensor]:
    """Yields shuffled item-index batches indefinitely.

    Args:
        num_items: Dataset size ``N``.
        batch_size: Items per batch, or None for full batch.
        seed: RNG seed for the shuffle.

    Yields:
        tf.Tensor: Int32 index vectors, length ``batch_size``.

    Note:
        Epochs are reshuffled and the trailing partial batch is dropped, so
        every yielded batch has the same length. That keeps ``tf.function``
        from retracing on the last batch of each epoch -- retracing is correct
        but expensive, and doing it once per epoch is a common, invisible
        source of slow training.
    """
    rng = np.random.default_rng(seed)

    if batch_size is None or batch_size >= num_items:
        full = tf.constant(np.arange(num_items), dtype=tf.int32)
        while True:
            yield full

    while True:
        perm = rng.permutation(num_items)
        for start in range(0, num_items - batch_size + 1, batch_size):
            yield tf.constant(perm[start : start + batch_size], dtype=tf.int32)


def train(
    model: GPCrowdModel,
    X: np.ndarray,
    labels: CrowdLabels,
    iterations: int = 2000,
    warmup_iterations: int = 0,
    batch_size: int | None = None,
    learning_rate: float = 0.01,
    seed: int = 0,
    compile_graph: bool = True,
    callback: Callable[[int, float], None] | None = None,
    early_stopping: bool = False,
    patience: int = 100,
    min_delta: float = 1e-4,
    smoothing_window: int = 10,
) -> TrainingHistory:
    """Fits the model by maximising the ELBO with Adam.

    Args:
        model: The composed model.
        X: Feature matrix. Shape ``[N, D]``.
        labels: The annotations.
        iterations: Number of optimiser steps in the main phase. With
            `early_stopping` on, this is only the *upper bound* -- training
            may stop sooner. ``len(history.elbo)`` is the number actually run.
        warmup_iterations: Optional number of warm-up steps run first, in
            which ``model.latent.gp_hyperparameters()`` (kernel hyperparameters
            and inducing-point locations -- ``Θ`` in the paper) is held frozen
            and only the remaining variational parameters (`q(Z)`, `q(U)`, the
            annotator's own parameters) are fitted. This is Algorithm 1's
            two-phase schedule: it lets ``q(Z)`` and the annotator model settle
            on a sensible labelling before the GP starts chasing it, which
            matters most for an annotator strategy with no data-driven
            initialisation of its own (e.g. `FeatDepDirichletAnnotator`, whose
            confusion matrices come from a freshly-initialised network rather
            than a seedable ``alpha_tilde``). 0 (default) skips warm-up
            entirely, matching every call site written before this parameter
            existed.
        batch_size: Items per step, or None for full batch.
        learning_rate: Adam learning rate.
        seed: Shuffling seed.
        compile_graph: Wrap the step in ``tf.function``. Leave True normally;
            set False when debugging, since eager execution gives readable
            tracebacks and lets you print intermediate tensors.
        callback: Optional ``(iteration, elbo)`` hook, for progress reporting.
            Iteration numbers run continuously across both phases: warm-up
            occupies ``0 .. warmup_iterations - 1``, and the main phase
            continues from there.
        early_stopping: Let the *main* phase stop before ``iterations`` once
            its ELBO has converged, rather than always running the full
            budget -- see `_EarlyStopper`. Warm-up always runs its full fixed
            length: Algorithm 1 gives it no convergence criterion of its own,
            it exists to prepare the variational parameters for a fixed
            number of steps before ``Θ`` unfreezes. Off by default so
            existing call sites keep training for exactly ``iterations``
            steps.
        patience: Consecutive non-improving checks allowed before stopping.
            Only used when `early_stopping` is True.
        min_delta: Minimum *relative* improvement in the smoothed ELBO that
            counts as still improving (e.g. ``1e-4`` = 0.01%). Only used when
            `early_stopping` is True.
        smoothing_window: Number of iterations averaged over before comparing
            to the best value seen so far, to smooth out per-minibatch noise.
            Only used when `early_stopping` is True.

    Returns:
        TrainingHistory: The ELBO and component traces for both phases,
        warm-up first. ``len(history.elbo)`` is the total number of steps
        actually run, including warm-up -- with `early_stopping` on, this can
        be less than ``warmup_iterations + iterations``.

    Note:
        ``model.trainable_variables`` is collected fresh inside the step rather
        than captured once, because ``gpflow.Module`` discovers parameters by
        traversing attributes. Capturing the list outside would silently miss
        any component attached after construction. The frozen-``Θ`` variable
        set is excluded by identity (``id()``) from that same fresh list on
        every warm-up step, rather than toggled via GPflow's ``trainable``
        flag, so nothing needs to be restored afterwards and a failure
        mid-warm-up cannot leave the model's parameters stuck non-trainable.

        Converting each term to ``float`` forces a device sync every iteration.
        Fine at this scale, and worth the cost for the diagnostics; if you
        later train on GPU with thousands of fast steps, accumulate on device
        and transfer periodically instead.

        Each phase gets its own ``Adam`` instance rather than sharing one:
        Keras' current optimizer registers the exact variable list it is
        first called with, and raises if a later call passes a variable
        (e.g. a re-enabled ``Θ``) it has not seen before -- "the optimizer
        cannot recognize variable...". A fresh optimizer per phase keeps each
        one's variable set constant for its own lifetime, at the cost of
        Adam's moment estimates resetting at the warm-up/main boundary. That
        reset is harmless here: Algorithm 1 does not require continuity
        across the two phases either.
    """
    X_tf = tf.constant(np.asarray(X, dtype=np.float64))
    history = TrainingHistory()

    def make_step(excluded_ids: frozenset[int]):
        opt = tf.optimizers.Adam(learning_rate)

        def step(idx: tf.Tensor):
            batch = labels.gather_batch(X_tf, idx)
            with tf.GradientTape() as tape:
                terms = model.elbo_terms(batch)
                loss = -terms.total
            variables = [v for v in model.trainable_variables if id(v) not in excluded_ids]
            opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return terms

        return tf.function(step) if compile_graph else step

    def record(terms, iteration: int) -> float:
        elbo_value = float(terms.total)
        history.elbo.append(elbo_value)
        history.latent.append(float(terms.latent))
        history.crowd.append(float(terms.crowd))
        history.entropy.append(float(terms.entropy))
        history.kl_latent.append(float(terms.kl_latent))
        history.kl_annotator.append(float(terms.kl_annotator))
        if callback is not None:
            callback(iteration, elbo_value)
        return elbo_value

    if warmup_iterations > 0:
        theta_ids = frozenset(id(v) for v in model.latent.gp_hyperparameters())
        warmup_step = make_step(theta_ids)
        batches = zip(range(warmup_iterations), batch_iterator(labels.num_items, batch_size, seed))
        for i, idx in batches:
            record(warmup_step(idx), i)

    stopper = _EarlyStopper(patience, min_delta, smoothing_window) if early_stopping else None
    main_step = make_step(frozenset())
    batches = zip(range(iterations), batch_iterator(labels.num_items, batch_size, seed))
    for i, idx in batches:
        elbo_value = record(main_step(idx), warmup_iterations + i)
        if stopper is not None and stopper.should_stop(elbo_value):
            break

    return history