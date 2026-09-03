"""Concrete annotator strategies.

Three interchangeable worker-noise models, all satisfying the
[AnnotatorModel][gpcrowdkit.annotators.base.AnnotatorModel] object. The core engine
never learns which one it holds, so adding a fourth requires no change
anywhere else in the library.

They span the useful range of complexity:

============================            =========================  ==============================
Strategy                                Parameters                 Worker uncertainty
============================            =========================  ==============================
VariationalDirichletAnnotator           ``A * C * C``              full posterior over ``R``
SoftmaxPointAnnotator                   ``A * C * C``              none (point estimate)
OneCoinAnnotator                        ``A``                      none (point estimate)
FeatDepDirichletAnnotator               ``O(hidden)``, shared      full posterior over ``R(X)``
                                         across ``A`` and ``C``
============================            =========================  ==============================

[OneCoinAnnotator][gpcrowdkit.annotators.strategies.OneCoinAnnotator] is the one that justifies the shape of the base
class. It carries a single scalar per worker, so an interface promising an
``[A, C, C]`` parameter tensor would have been the wrong abstraction -- it
happens to *build* such a tensor, but it does not store one.

Index convention throughout: tensors are ``[A, C_obs, C_true]``, normalised
down axis 1, so each column is one distribution over observed labels given a
fixed true class.
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive

from ..data import CrowdBatch, CrowdLabels
from .base import AnnotatorModel, ConfusionAnnotator

__all__ = [
    "VariationalDirichletAnnotator",
    "SoftmaxPointAnnotator",
    "OneCoinAnnotator",
    "FeatDepDirichletAnnotator",
    "FeatDepVariationalDirichletAnnotator",
    "init_alpha_tilde",
    "ALL_ANNOTATOR_STRATEGIES",
]

FLOAT = tf.float64


class VariationalDirichletAnnotator(ConfusionAnnotator):
    """Full variational Dirichlet posterior over confusion matrices (SVGPCR).

    Places an independent Dirichlet on each *column* of each worker's confusion
    matrix -- one distribution over observed labels per true class::

        q(R^a_{.,j}) = Dir(alpha_tilde^a_{.,j})
        p(R^a_{.,j}) = Dir(alpha^a_{.,j})

    This is the strategy from SVGPCR:
    
    Morales-Alvarez, P., Ruiz, P., Coughlin, S., Molina, R., & Katsaggelos, A. K. (2022). 
    Scalable Variational Gaussian Processes for Crowdsourcing: Glitch Detection in LIGO. 
    IEEE transactions on pattern analysis and machine intelligence, 44(3), 1534-1551.
    
    The only one here
    that represents *uncertainty* about a worker rather than a best guess. That
    matters when workers are sparse: a worker with three annotations and a
    worker with three thousand can have identical point estimates but wildly
    different posteriors, and only this strategy can tell them apart.

    Attributes:
        alpha (gpflow.Parameter): Prior concentrations, non-trainable.
            Shape ``[A, C_obs, C_true]``.
        alpha_tilde (gpflow.Parameter): Variational concentrations, trainable,
            constrained positive. Shape ``[A, C_obs, C_true]``.
    """

    def __init__(
        self,
        num_workers: int,
        num_classes: int,
        alpha_prior: np.ndarray | float = 1.0,
        alpha_tilde_init: np.ndarray | None = None,
    ) -> None:
        """Initialises the prior and variational Dirichlet concentrations.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            alpha_prior: Prior concentrations: a scalar broadcast over
                ``[A, C, C]``, or an explicit array of that shape. A flat 1.0
                (the reference implementation's choice) is the uniform
                Dirichlet -- no prior opinion about any worker.
            alpha_tilde_init: Optional ``[A, C, C]`` starting point. Defaults to
                a mild diagonal bias, encoding the assumption that workers are
                better than chance. See [init_alpha_tilde][gpcrowdkit.annotators.strategies.init_alpha_tilde] for the
                data-driven alternative, which converges considerably faster.

        Note:
            ``alpha`` is stored as a non-trainable ``Parameter`` rather than a
            ``tf.constant`` so that it appears in ``gpflow.utilities.print_summary``
            alongside everything else, and so that a subclass can make the
            prior learnable (empirical Bayes) by flipping one flag.
        """
        super().__init__(num_workers, num_classes)
        shape = (self.A, self.C, self.C)
        # Adapt the alpha_prior to the shape and type required by the flow.
        prior = np.broadcast_to(np.asarray(alpha_prior, dtype=np.float64), shape).copy()
        # Set the prior as a gpflow parameter and lock it to non-trainable.
        self.alpha = gpflow.Parameter(prior, transform=positive(), trainable=False)

        # By default, initialise the annotators to be better than random (1+1/C on the diagonal, 1/C off-diagonal).
        if alpha_tilde_init is None:
            alpha_tilde_init = np.full(shape, 1.0 / self.C) + np.stack(
                [np.eye(self.C) for _ in range(self.A)]
            )
        self.alpha_tilde = gpflow.Parameter(
            np.asarray(alpha_tilde_init, dtype=np.float64), transform=positive()
        )

    def expected_log_confusion(self) -> tf.Tensor:
        """``E_q[log R] = psi(alpha_tilde) - psi(sum_i alpha_tilde_{i,j})``.

        The standard Dirichlet identity. Note this is an expectation of a
        logarithm, computed exactly -- not the logarithm of the mean, which
        would be a different and biased quantity.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.
        """
        a = self.alpha_tilde
        return tf.math.digamma(a) - tf.math.digamma(tf.reduce_sum(a, axis=1, keepdims=True))

    def kl_divergence(self) -> tf.Tensor:
        """Analytic Dirichlet-Dirichlet KL, summed over workers and columns.

        For a single column, ``KL(Dir(q) || Dir(p))`` is::

            log B(p) - log B(q) + sum_i (q_i - p_i) (psi(q_i) - psi(sum q))

        The implementation evaluates all ``A * C`` columns at once by
        distributing that expression across the tensor.

        Returns:
            tf.Tensor: Scalar, non-negative, and exactly zero when
            ``alpha_tilde == alpha``.

        Note:
            ``tf.math.lbeta`` reduces the *last* axis, but the Dirichlet lives
            along axis 1 (observed classes). ``matrix_transpose`` swaps the last
            two axes so that the reduction lands on the right one. Omitting it
            computes a log-beta over true classes instead: still finite, still
            differentiable, silently wrong.
        """
        q, p = self.alpha_tilde, self.alpha
        diff = q - p
        term1 = tf.reduce_sum(diff * tf.math.digamma(q))
        term2 = -tf.reduce_sum(
            tf.math.digamma(tf.reduce_sum(q, axis=1)) * tf.reduce_sum(diff, axis=1)
        )
        term3 = tf.reduce_sum(
            tf.math.lbeta(tf.linalg.matrix_transpose(p))
            - tf.math.lbeta(tf.linalg.matrix_transpose(q))
        )
        return term1 + term2 + term3

    def confusion_matrices(self) -> tf.Tensor:
        """Posterior mean ``E_q[R] = alpha_tilde / sum_i alpha_tilde_{i,j}``.

        The Dirichlet is the normalised vector of independent Gammas, so its
        mean is each concentration over their sum.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.

        Note:
            Deliberately *not* ``softmax(expected_log_confusion())``. That is
            the normalised geometric mean, which is more sharply peaked than
            the posterior mean and so overstates worker accuracy -- by around
            0.06 on the diagonal at ``alpha = [10, 1, 1]``, and more as the
            concentrations shrink. The bias is therefore largest exactly where
            data is scarce and the estimate matters most.
        """
        return self.alpha_tilde / tf.reduce_sum(self.alpha_tilde, axis=1, keepdims=True)


class SoftmaxPointAnnotator(ConfusionAnnotator):
    """Deterministic point-estimate confusion matrices, ``R^a = softmax(W^a)``.

    Same parameter count as the Dirichlet strategy but no distribution over
    them, so no KL term and no notion of how confident the estimate is. Cheaper
    and often adequate when every worker has plenty of annotations.

    Attributes:
        logits (gpflow.Parameter): Unconstrained logits. Shape ``[A, C, C]``.
    """

    def __init__(self, num_workers: int, num_classes: int, diagonal_init: float = 2.0) -> None:
        """Initialises the logits with a diagonal bias.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            diagonal_init: Logit mass on the diagonal at initialisation.
                Softmax of ``2.0`` on the diagonal gives a competent-but-not-
                certain worker, a reasonable neutral start.

        Note:
            No constraint transform is needed: the softmax in
            [expected_log_confusion][gpcrowdkit.annotators.base.ConfusionAnnotator.expected_log_confusion] handles normalisation, so the raw
            logits are free parameters over all of R.
        """
        super().__init__(num_workers, num_classes)
        init = np.tile(np.eye(num_classes) * diagonal_init, (num_workers, 1, 1))
        self.logits = gpflow.Parameter(init.astype(np.float64))

    def expected_log_confusion(self) -> tf.Tensor:
        """``log R = log_softmax(logits)`` down the observed-class axis.

        Exact rather than an expectation, since ``R`` is deterministic here.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.

        Note:
            ``log_softmax`` rather than ``log(softmax(...))``: it subtracts the
            max internally, so it stays finite for logits where the naive
            composition would produce ``log(0) = -inf``.
        """
        return tf.nn.log_softmax(self.logits, axis=1)

    def kl_divergence(self) -> tf.Tensor:
        """Zero: a point estimate carries no distribution to penalise."""
        return tf.constant(0.0, dtype=FLOAT)

    def confusion_matrices(self) -> tf.Tensor:
        """Exact confusion matrices, ``softmax(logits)``.

        No expectation is involved, so unlike the Dirichlet case this really is
        the exponential of [expected_log_confusion][gpcrowdkit.annotators.base.ConfusionAnnotator.expected_log_confusion].

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        return tf.nn.softmax(self.logits, axis=1)


class OneCoinAnnotator(ConfusionAnnotator):
    """Single scalar accuracy per worker -- the classic one-coin model.

    Worker ``a`` is correct with probability ``beta_a`` and otherwise spreads
    the remaining mass uniformly::

        R^a_{ij} = beta_a               if i == j
                   (1 - beta_a)/(C-1)   otherwise

    Strong assumption -- it cannot express a worker who systematically confuses
    two particular classes -- but with ``A`` parameters instead of ``A*C*C`` it
    is far better behaved when annotations per worker are scarce, which is the
    usual situation.

    It is also the design test for the base contract: it stores ``A`` scalars,
    not a confusion tensor. An interface that had promised ``[A, C, C]``
    parameters would have been the wrong abstraction.

    Attributes:
        beta_logit (gpflow.Parameter): Unconstrained per-worker accuracy
            logits. Shape ``[A]``.
    """

    def __init__(self, num_workers: int, num_classes: int, init_accuracy: float = 0.7) -> None:
        """Initialises per-worker accuracies.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``, at least 2.
            init_accuracy: Initial accuracy shared by all workers, in ``(0, 1)``.

        Raises:
            ValueError: If ``num_classes < 2`` (the off-diagonal mass would be
                divided by zero), or if ``init_accuracy`` is not in ``(0, 1)``.

        Note:
            Stored as a logit with the sigmoid applied in [beta][gpcrowdkit.annotators.strategies.OneCoinAnnotator.beta], rather
            than as a constrained ``Parameter``. Both work; the logit keeps the
            dependency surface small and makes the unconstrained optimisation
            explicit at the point of use.
        """
        if num_classes < 2:
            raise ValueError("OneCoinAnnotator requires at least two classes.")
        if not 0.0 < init_accuracy < 1.0:
            raise ValueError("init_accuracy must lie strictly in (0, 1).")
        super().__init__(num_workers, num_classes)
        logit = float(np.log(init_accuracy / (1.0 - init_accuracy)))
        self.beta_logit = gpflow.Parameter(np.full(num_workers, logit, dtype=np.float64))

    @property
    def beta(self) -> tf.Tensor:
        """Per-worker accuracy in ``(0, 1)``. Shape ``[A]``."""
        return tf.sigmoid(self.beta_logit)

    def _confusion(self) -> tf.Tensor:
        """Builds the ``[A, C, C]`` confusion tensor from the ``A`` scalars.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        beta = self.beta[:, None, None]  # [A, 1, 1]
        eye = tf.eye(self.C, dtype=FLOAT)[None, :, :]  # [1, C, C]
        off = (1.0 - beta) / tf.constant(self.C - 1, dtype=FLOAT)
        return eye * beta + (1.0 - eye) * off

    def expected_log_confusion(self) -> tf.Tensor:
        """Log of the constructed confusion tensor.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.

        Note:
            The ``1e-12`` floor guards ``log(0)`` if the sigmoid saturates at
            0 or 1 during optimisation. Saturation is itself a symptom -- a
            worker driven to perfect accuracy usually means too few
            annotations and no prior to regularise them.
        """
        return tf.math.log(self._confusion() + 1e-12)

    def kl_divergence(self) -> tf.Tensor:
        """Zero: a point estimate carries no distribution to penalise."""
        return tf.constant(0.0, dtype=FLOAT)

    def confusion_matrices(self) -> tf.Tensor:
        """Exact confusion matrices built from the per-worker accuracies.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        return self._confusion()


def init_alpha_tilde(
    labels: CrowdLabels, class_probs: np.ndarray, prior_strength: float = 1.0
) -> np.ndarray:
    """Data-driven initialisation of the variational Dirichlet concentrations.

    Reproduces ``_init_behaviors`` from the reference implementation: each
    annotation ``(n, a, y)`` adds the soft class assignment ``class_probs[n]``
    to ``alpha_tilde[a, y, :]``, so a worker who labels an item "cat" when the
    votes point to "cat" accumulates diagonal mass.

    Starting here rather than from a flat prior matters. The ELBO is not convex,
    and from a uniform start the model can settle into a labelling that is a
    permutation of the truth -- self-consistent, high-likelihood, and useless.
    Vote-based initialisation breaks that symmetry before optimisation begins.

    Args:
        labels: The annotations.
        class_probs: Soft per-item class assignments ``[N, C]``, typically
            [empirical_class_probs][gpcrowdkit.data.CrowdLabels.empirical_class_probs].
        prior_strength: Constant added to every entry, keeping concentrations
            comfortably positive and the digamma well conditioned.

    Returns:
        np.ndarray: Shape ``[A, C_obs, C_true]``, float64.

    Note:
        The two nested Python loops of the original become two ``np.add.at``
        scatter-adds, turning an ``O(L)`` interpreter loop into two vectorised
        passes. On a dataset with millions of annotations this is the
        difference between minutes and seconds of setup.
    """
    A, C = labels.num_workers, labels.num_classes
    acc = np.full((A, C, C), 1.0 / C)
    counts = np.ones((A, C))

    np.add.at(acc, (labels.worker_idx, labels.label), class_probs[labels.item_idx])
    np.add.at(counts, (labels.worker_idx, labels.label), 1.0)

    acc /= counts[:, :, None]
    acc *= (counts / counts.sum(axis=1, keepdims=True))[:, :, None]
    acc /= acc.sum(axis=1, keepdims=True)
    return acc + prior_strength


def _inverse_softplus(x: np.ndarray) -> np.ndarray:
    """Inverts ``softplus``: returns ``y`` such that ``log1p(exp(y)) == x``.

    Used to seed `FeatDepDirichletAnnotator`'s baseline table so that
    ``softplus(baseline) + 1e-3`` reproduces a target concentration array
    (typically [init_alpha_tilde][gpcrowdkit.annotators.strategies.init_alpha_tilde]'s output) exactly at
    initialisation. ``log(expm1(x))`` rather than ``log(exp(x) - 1)``: `expm1`
    avoids the catastrophic cancellation of computing ``exp(x) - 1`` directly
    for the modest, near-1 concentrations `init_alpha_tilde` produces.

    Args:
        x: Positive array, shape arbitrary.

    Returns:
        np.ndarray: Same shape as ``x``.
    """
    return np.log(np.expm1(np.maximum(x, 1e-6)))


class FeatDepDirichletAnnotator(AnnotatorModel):
    """ Variational Dirichlet Strategy with feature-dependent confusion matrices.

    note: It inherits from AnnotatorModel instead of ConfusionAnnotator because
    the confusion matrices are now dependent on the input features X, and the implementation
    doesn't match the structure of ConfusionAnnotator.

    Per the paper's Eq. 2 ("a neural network which receives the annotator,
    features, and true class... and produces K values"), the network is a
    single shared trunk taking ``(x_n, annotator, true class)`` as explicit
    inputs and returning the ``C`` concentrations of *one* Dirichlet column --
    not, as a network that only ever saw ``X`` would have to, a private
    ``A * C * C``-sized output block computed once per item and sliced by
    annotator. Annotator identity is a small trainable embedding rather than
    a one-hot (the paper does not specify which; an embedding keeps the
    parameter count from scaling with ``A`` and lets the model discover
    annotators with similar behaviour); true class is a one-hot, since ``C``
    is fixed and small. Every ``(annotator, true class)`` combination is
    evaluated in one batched forward pass -- ``get_alpha_tilde`` below -- so
    this stays compatible with ``tf.function`` the same way
    [AnnotatorModel.crowd_log_per_item][gpcrowdkit.annotators.base.AnnotatorModel.crowd_log_per_item]'s
    vectorised segment-sum avoids a Python loop over annotators.

    ``alpha_tilde(x, a, j)`` is a residual: a per-``(a, j)`` trainable
    baseline table, initialised (like `VariationalDirichletAnnotator`'s own
    `alpha_tilde`) from `init_alpha_tilde`, plus a correction from the shared
    trunk above whose *last layer* starts at exactly zero. At step 0 the
    correction contributes nothing, so this strategy starts numerically
    identical to `VariationalDirichletAnnotator` -- vote-based, ``X``-free --
    and only differentiates by ``X`` as training moves the trunk's weights
    away from zero. Without this, the trunk starts from Keras' default random
    init: a flat, uninformative Dirichlet column for every annotator and
    class, independent of ``X``, which is exactly the symmetric starting
    point `init_alpha_tilde`'s docstring warns can settle into "a labelling
    that is a permutation of the truth".

    ``x`` itself is not fed to the trunk directly: a linear, no-activation
    ``feature_proj`` layer first maps it ``D -> feature_bottleneck_dim``. Almost
    the entire parameter count of a naive version of this strategy comes from
    exactly this step -- the trunk's first hidden layer's ``D``-columns block,
    where ``D`` is whatever a foundation-model backbone happens to embed items
    in (512-1024 for AI4SkINv2-2's backbones), not anything intrinsic to how
    complex "annotator reliability as a function of item content" actually is.
    Low-rank-factorising that block into ``D -> r -> hidden`` instead of
    ``D -> hidden`` directly cuts parameters roughly in proportion to
    ``r / hidden_units[0]`` without touching what the trunk can subsequently
    do with ``(x, annotator, class)`` -- and because ``feature_proj`` is
    trained end-to-end through the same ELBO gradient as everything else, it
    orients itself toward whichever directions of ``X`` the data says predict
    annotator reliability, rather than (say) a fixed PCA's variance-maximising
    ones. It is also computed once per item and *then* tiled across every
    ``(annotator, class)`` combination (see `get_alpha_tilde`), rather than
    tiling ``x`` first: the projection no longer does ``A * C`` redundant
    copies of the same computation.
    """

    def __init__(
        self,
        num_workers: int,
        num_classes: int,
        X: np.ndarray,
        hidden_units: list[int] = [64, 64],
        annotator_embedding_dim: int = 8,
        feature_bottleneck_dim: int | None = 1,
        alpha_prior: float = 1.0,
        alpha_tilde_init: np.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        """Initialises the feature-dependent Dirichlet annotator.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            X: Full training feature matrix, shape ``[N, D]``. Only its row
                count ``N`` is kept (not the array itself): `kl_divergence`
                needs it to rescale a batch-only estimate up to a full-dataset
                one, the same ``N/B`` correction
                [ELBOTerms.total][gpcrowdkit.models.ELBOTerms.total] already
                applies to the ``latent``/``crowd``/``entropy`` terms.
            hidden_units: Hidden layer widths of the shared
                ``(x_proj, annotator, true class) -> alpha_tilde column`` trunk,
                run on the *projected* features -- see `feature_bottleneck_dim`.
            annotator_embedding_dim: Width of the trainable annotator
                embedding fed to the trunk alongside ``x_proj`` and the
                one-hot true class.
            feature_bottleneck_dim: Width ``r`` of the linear, no-activation
                projection ``D -> r`` applied to ``x`` before it reaches the
                trunk. See the class docstring for why this is where most of
                this strategy's parameters would otherwise go, and why
                shrinking it is expected to cost little to no expressiveness
                for this task specifically. Silently capped at ``D`` (the
                actual feature width): a "bottleneck" wider than the input it
                bottlenecks would expand rather than shrink that block --
                harmless on real foundation-model features (``D`` in the
                hundreds), but exactly what would happen by default on a
                toy 2-D dataset otherwise. Pass ``None`` to disable the
                bottleneck entirely and feed raw ``x`` to the trunk, the
                behaviour before this parameter existed -- an easy way to
                rule it out when comparing against or debugging that earlier
                behaviour, without deleting `feature_proj` or anything that
                uses it.
            alpha_prior: Prior concentration, broadcast over ``[A, C, C]``.
            alpha_tilde_init: Optional ``[A, C, C]`` starting point for the
                ``X``-independent baseline table, in the same shape and
                convention as `VariationalDirichletAnnotator`'s own parameter
                of the same name -- typically
                [init_alpha_tilde][gpcrowdkit.annotators.strategies.init_alpha_tilde],
                so both strategies can be seeded from the same call. Defaults
                to the same mild diagonal bias `VariationalDirichletAnnotator`
                falls back to when unset.
            name: Optional module name, forwarded to ``gpflow.Module``.
        """
        super().__init__(num_workers, num_classes, name=name)

        # Non-trainable alpha prior [1,A,C,C] for broadcasting over batch B.
        prior_array = np.full((1, self.A, self.C, self.C), alpha_prior, dtype=np.float64)
        self.alpha = gpflow.Parameter(prior_array, transform=positive(), trainable=False)

        # Linear D -> r bottleneck, trained jointly with everything else. See the
        # class docstring: this low-rank-factorises what would otherwise be the
        # trunk's first layer's D-columns block, its largest piece by far.
        # Capped at D so it can only ever shrink, never expand, that block.
        # feature_bottleneck_dim=None turns it into a plain pass-through (tf.identity
        # has no parameters and isn't a tf.Module, so it contributes nothing to
        # trainable_variables) -- an easy toggle back to feeding raw x to the trunk,
        # with nothing to delete or re-add either way.
        if feature_bottleneck_dim is None:
            self.feature_proj = tf.identity
        else:
            feature_dim = int(np.asarray(X).shape[1])
            feature_bottleneck_dim = min(feature_bottleneck_dim, feature_dim)
            self.feature_proj = tf.keras.layers.Dense(feature_bottleneck_dim, dtype=FLOAT)

        # Annotator identity is a learned embedding; true class is one-hot inside
        # get_alpha_tilde (C is fixed and small, unlike A which can be large).
        self.annotator_embed = tf.keras.layers.Embedding(self.A, annotator_embedding_dim, dtype=FLOAT)

        # Shared trunk: (x_proj, annotator_embedding, true_class_onehot) -> the C
        # concentrations of that one Dirichlet column, as a *correction* added
        # to the data-driven baseline built below. The last layer's kernel and
        # bias are zero-initialised -- the standard "zero-init the residual
        # branch" trick -- so this correction is exactly 0 for every input at
        # step 0 regardless of the (randomly initialised) hidden layers, the
        # embedding, or feature_proj: only the baseline determines alpha_tilde
        # initially.
        layers = []
        for units in hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation="relu", dtype=FLOAT))
        layers.append(tf.keras.layers.Dense(
            self.C, dtype=FLOAT, kernel_initializer="zeros", bias_initializer="zeros",
        ))
        self.net = tf.keras.Sequential(layers)

        # Precomputed (annotator, true_class) combination pattern shared by every
        # item: length A*C, combo k = a*C + j. See get_alpha_tilde().
        self._ann_per_combo = tf.constant(np.repeat(np.arange(self.A), self.C), dtype=tf.int32)
        self._class_per_combo = tf.constant(np.tile(np.arange(self.C), self.A), dtype=tf.int32)
        self._combo_idx = tf.constant(np.arange(self.A * self.C), dtype=tf.int32)

        # X-independent, per-(annotator, true class) baseline, trainable so it
        # keeps refining beyond its initial value -- effectively this
        # strategy's own analogue of VariationalDirichletAnnotator.alpha_tilde,
        # with the trunk above contributing whatever X-dependent correction
        # the data supports on top of it.
        if alpha_tilde_init is None:
            alpha_tilde_init = np.full((self.A, self.C, self.C), 1.0 / self.C) + np.stack(
                [np.eye(self.C) for _ in range(self.A)]
            )
        baseline_logits = _inverse_softplus(np.asarray(alpha_tilde_init, dtype=np.float64) - 1e-3)
        # [A, C_obs, C_true] -> [A, C_true, C_obs] -> [A*C, C_obs], combo k = a*C+j.
        baseline_logits = np.transpose(baseline_logits, (0, 2, 1)).reshape(self.A * self.C, self.C)
        self._alpha_tilde_baseline = tf.keras.layers.Embedding(
            self.A * self.C, self.C, dtype=FLOAT,
            embeddings_initializer=tf.keras.initializers.Constant(baseline_logits),
        )

        # Dataset size, for the N/B rescaling in kl_divergence() -- see there.
        self._num_items = int(np.asarray(X).shape[0])
        # Cache of the most recent batch's X, used by kl_divergence() and by
        # confusion_matrices() for reporting.
        self._last_X: tf.Tensor | None = None

    def get_alpha_tilde(self, X: tf.Tensor) -> tf.Tensor:
        """Returns alpha_tilde(X) with shape [B, A, C_obs, C_true].

        Projects ``X`` to `feature_bottleneck_dim` *once*, then builds one row
        of ``(x_proj, annotator_embedding, true_class_onehot)`` per item per
        ``(annotator, true class)`` combination -- ``B * A * C`` rows total --
        and runs all of them through the shared trunk in a single batched
        call, rather than looping over ``A`` and ``C`` in Python (or
        recomputing the ``D -> r`` projection redundantly for every combo, by
        projecting before tiling instead of after). The trunk's output is
        added to the data-driven, ``X``-independent baseline (see the class
        docstring) rather than used on its own.
        """
        B = tf.shape(X)[0]
        AC = self.A * self.C

        x_proj = self.feature_proj(X)  # [B, r] -- the expensive D-dim step, done once
        x_proj_tiled = tf.repeat(x_proj, repeats=AC, axis=0)  # [B*A*C, r]
        ann_ids = tf.tile(self._ann_per_combo, [B])  # [B*A*C]
        true_class = tf.tile(self._class_per_combo, [B])  # [B*A*C]
        combo_ids = tf.tile(self._combo_idx, [B])  # [B*A*C]

        ann_embed = self.annotator_embed(ann_ids)  # [B*A*C, E]
        class_onehot = tf.one_hot(true_class, self.C, dtype=FLOAT)  # [B*A*C, C]

        net_in = tf.concat([x_proj_tiled, ann_embed, class_onehot], axis=-1)
        correction = self.net(net_in)  # [B*A*C, C_obs], ~0 at initialisation
        baseline = self._alpha_tilde_baseline(combo_ids)  # [B*A*C, C_obs], X-independent
        raw_out = baseline + correction  # the K values of Eq. 2

        reshaped = tf.reshape(raw_out, (B, self.A, self.C, self.C))  # [B, A, C_true, C_obs]
        reshaped = tf.transpose(reshaped, perm=[0, 1, 3, 2])  # [B, A, C_obs, C_true]
        return tf.nn.softplus(reshaped) + 1e-3

    def expected_log_confusion_all(self, X: tf.Tensor) -> tf.Tensor:
        """
        Expected log-confusion: E_q[log(R|X)], with shape [B, A, C, C]
        """
        a = self.get_alpha_tilde(X)
        return tf.math.digamma(a) - tf.math.digamma(tf.reduce_sum(a, axis=2, keepdims=True))

    def label_log_terms(self, batch: CrowdBatch) -> tf.Tensor:
        """
        Base Contract: Returns [L,C] for each one of the L annotations of the batch
        """
        X = batch.X
        self._last_X = X  # Guardar referencia para kl_divergence()

        # Matriz completa log-esperada para el lote: [B, A, C_obs, C_true]
        E_log_R = self.expected_log_confusion_all(X)

        # Mapear L anotaciones a (índice local del ítem [0..B-1], anotador, etiqueta observada)
        idx = tf.stack([batch.item_local, batch.worker_idx, batch.label], axis=-1)  # [L, 3]

        # Extraer el vector sobre las C_true clases candidatas -> [L, C_true]
        return tf.gather_nd(E_log_R, idx)

    def kl_divergence(self) -> tf.Tensor:
        """``KL(q(R|X) || p(R))``, estimated from the current batch and rescaled to the full dataset.

        Unlike `VariationalDirichletAnnotator`'s KL, which lives on a fixed
        ``[A, C, C]`` tensor and is genuinely complete regardless of batch,
        ``alpha_tilde`` here is a function of each item's own ``x_n`` -- so the
        *true* KL is itself a sum over all ``N`` items, exactly like the
        ``latent``/``crowd``/``entropy`` terms `GPCrowdModel.elbo_terms` computes.
        `ELBOTerms.total` doesn't know that: it always adds `kl_annotator`
        unscaled, on the assumption -- correct for the global-parameter
        strategies -- that it is already complete. Rather than making that true
        by evaluating the network over every training item on every call (an
        earlier version of this method did exactly that), this reproduces the
        *effect* of the model's own ``N/B`` correction from inside the
        strategy itself: sum the KL over the batch `label_log_terms` already
        computed (`self._last_X`, no extra forward pass needed) and scale by
        ``N/B``, the identical factor `ELBOTerms.total` applies to the data
        terms. This is the standard stochastic-VI treatment of a per-item KL
        under minibatching -- an unbiased estimate of the full sum on every
        call, not the exact value, in exchange for ``O(B)`` instead of
        ``O(N)`` work per step.

        Returns:
            tf.Tensor: Scalar.
        """
        if self._last_X is None:
            return tf.constant(0.0, dtype=FLOAT)

        q = self.get_alpha_tilde(self._last_X)  # [B, A, C_obs, C_true]
        p = self.alpha                          # [1, A, C_obs, C_true]

        diff = q - p
        term1 = tf.reduce_sum(diff * tf.math.digamma(q))
        term2 = -tf.reduce_sum(
            tf.math.digamma(tf.reduce_sum(q, axis=2, keepdims=True))
            * tf.reduce_sum(diff, axis=2, keepdims=True)
        )

        q_trans = tf.transpose(q, perm=[0, 1, 3, 2])
        p_trans = tf.transpose(p, perm=[0, 1, 3, 2])
        term3 = tf.reduce_sum(tf.math.lbeta(p_trans) - tf.math.lbeta(q_trans))

        batch_kl = term1 + term2 + term3
        batch_size = tf.cast(tf.shape(self._last_X)[0], FLOAT)
        scale = tf.cast(self._num_items, FLOAT) / batch_size
        return batch_kl * scale

    def confusion_matrices(self, X: tf.Tensor | None = None) -> tf.Tensor:
        """
        Posterior mean of the confusion matrices: shape [B, A, C, C] (one matrix CxC per
        annotator and item in the batch). If X is None, uses the last batch's X.
        """
        target_X = X if X is not None else self._last_X
        if target_X is None:
            raise ValueError("Se requiere pasar X o haber ejecutado label_log_terms previamente.")
        a = self.get_alpha_tilde(target_X)
        return a / tf.reduce_sum(a, axis=2, keepdims=True)


ALL_ANNOTATOR_STRATEGIES = (
    VariationalDirichletAnnotator,
    SoftmaxPointAnnotator,
    OneCoinAnnotator,
    FeatDepDirichletAnnotator,
)

# Backward-compatible alias: some examples and docs use the longer, explicit name.
FeatDepVariationalDirichletAnnotator = FeatDepDirichletAnnotator