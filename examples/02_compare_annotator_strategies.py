"""Swap the annotator strategy without touching anything else.

`GPCrowdModel` never inspects which concrete `AnnotatorModel` it holds -- it
only calls the abstract contract in `gpcrowdkit.annotators.base`. This script is
the demonstration: the same synthetic dataset and the same latent GP are
reused across all three shipped strategies, and only the one line that
constructs `annotator=` changes.

Run with::

    ./run.sh examples/02_compare_annotator_strategies.py
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf

from gpcrowdkit import (
    FreeCategoricalZ,
    GPCrowdModel,
    OneCoinAnnotator,
    SVGPLatent,
    SoftmaxPointAnnotator,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
    make_synthetic,
    train,
)
from gpcrowdkit.annotators.base import AnnotatorModel


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


def build_annotator(cls: type[AnnotatorModel], labels, class_probs) -> AnnotatorModel:
    """The only place that knows which strategy is which."""
    if cls is VariationalDirichletAnnotator:
        return cls(
            labels.num_workers,
            labels.num_classes,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        )
    return cls(labels.num_workers, labels.num_classes)


def main() -> None:
    gpflow.config.set_default_float(np.float64)

    data = make_synthetic(
        num_items=300, num_features=2, num_classes=3, num_workers=10,
        labels_per_item=3, good_worker_fraction=0.5, seed=1,
    )
    labels = data.labels
    class_probs = labels.empirical_class_probs()
    mv_acc = accuracy(labels.majority_vote(), data.z)

    ## Strategies: - VariationalDirichletAnnotator: 3 params per worker, 
    #                   returns a Dirichlet distribution over the confusion matrix of each worker.
    #             - SoftmaxPointAnnotator: 3 params per worker, returns a single point estimate of the confusion matrix of each worker.
    #             - OneCoinAnnotator: 1 param per worker: prob of being right. Errors distribute uniformly 
    #                       among the other classes.                                 

    strategies = [VariationalDirichletAnnotator, SoftmaxPointAnnotator, OneCoinAnnotator]
    print(f"{'strategy':<28s} {'params/worker':>14s} {'accuracy':>10s} {'final elbo':>12s}")
    print(f"{'majority vote (baseline)':<28s} {'-':>14s} {mv_acc:>10.3f} {'-':>12s}")

    ## For each strategy, build, train and evaluate a model. 
    for cls in strategies:
        model = GPCrowdModel(
            latent=SVGPLatent(
                kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0),
                num_classes=labels.num_classes,
                inducing_points=data.X[:20].copy(),
            ),
            annotator=build_annotator(cls, labels, class_probs),
            num_data=labels.num_items,
            q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
        )
        history = train(model, data.X, labels, iterations=300, learning_rate=0.05)
        acc = accuracy(model.infer_true_labels(tf.constant(data.X), labels), data.z)
        params_per_worker = sum(np.prod(v.shape) for v in model.annotator.trainable_variables) / (
            labels.num_workers or 1
        )
        print(f"\n{'strategy':<28s} {'params/worker':>14s} {'accuracy':>10s} {'final elbo':>12s}")
        print(f"{cls.__name__:<28s} {params_per_worker:>14.0f} {acc:>10.3f} {history.elbo[-1]:>12.2f}")
        ## Check the decomposition of the ELBO for each strategy.
        print(f"ELBO decomposition for {cls.__name__}, first vs last 10 iterations:")
        for name, series in [
            ("latent", history.latent),
            ("crowd", history.crowd),
            ("entropy", history.entropy),
            ("kl_latent", history.kl_latent),
            ("kl_annotator", history.kl_annotator),
            ("total_elbo", history.elbo) # Añadido para ver si mejora la suma promedio
        ]:
            print(f"  {name:12s} {np.mean(series[:10]):12.2f} -> {np.mean(series[-10:]):12.2f}")
    


if __name__ == "__main__":
    main()
