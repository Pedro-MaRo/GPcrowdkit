"""
Synthetic crowdsourcing dataset with feature-dependent annotator noise, based on the "moons" dataset or on the "circles" dataset.

Adapted to gpcrowdkit's native data format: a plain ``[N, D]`` feature matrix
plus a [CrowdLabels][gpcrowdkit.data.CrowdLabels] object holding the sparse
annotations in COO form (see ``gpcrowdkit.data`` for why that representation
was chosen over a padded ``[N, S, 2]`` array). The per-item ``(annotator,
label)`` pairs this generator naturally produces -- ``Y_cr``/``Y_mask`` below
-- are exactly the "reference SVGPCR input format" that
[CrowdLabels.from_pairs][gpcrowdkit.data.CrowdLabels.from_pairs] consumes, so building one from the other is a
straight masking operation with no reshaping needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from sklearn.datasets import make_circles, make_moons

from gpcrowdkit.data import CrowdLabels

__all__ = [
    "MoonsCrowdData",
    "load_moons_dataset",
    "load_circles_dataset",
    "load_config",
]


@dataclass
class MoonsCrowdData:
    """A generated moons/circles crowdsourcing dataset, in gpcrowdkit's format.

    Attributes:
        X_train (np.ndarray): Training features seen by both the crowd and the
            model. Shape ``[N, D]``.
        labels (CrowdLabels): Sparse crowd annotations over ``X_train`` -- the
            only supervision `GPCrowdModel` is allowed to see.
        z_train (np.ndarray): True training labels, hidden from the model and
            used only to score `infer_true_labels` and majority vote.
        X_test (np.ndarray): Held-out features with no annotations at all,
            used to check that the fitted latent GP generalises.
        z_test (np.ndarray): True labels for ``X_test``.
        n_classes (int): Number of classes ``C``.
        n_annotators (int): Number of annotators ``A``.
        class_names (list[str]): Human-readable class labels.
    """

    X_train: np.ndarray
    labels: CrowdLabels
    z_train: np.ndarray
    X_test: np.ndarray
    z_test: np.ndarray
    n_classes: int
    n_annotators: int
    class_names: list[str]


def _to_crowd_labels(Y_cr: np.ndarray, Y_mask: np.ndarray, num_items: int) -> CrowdLabels:
    """Converts the padded ``(Y_cr, Y_mask)`` annotation arrays to `CrowdLabels`.

    ``Y_cr[i, :, :]`` holds up to ``M`` ``(annotator, label)`` pairs for item
    ``i``, padded with ``Y_mask`` marking which slots are real. Dropping the
    padding per item recovers exactly the ragged ``Sequence[[S_n, 2]]`` layout
    [CrowdLabels.from_pairs][gpcrowdkit.data.CrowdLabels.from_pairs] expects.

    Args:
        Y_cr: Shape ``[N, M, 2]``, ``(annotator_id, label)`` pairs.
        Y_mask: Shape ``[N, M]``, True where the slot holds a real annotation.
        num_items: Dataset size ``N``.

    Returns:
        CrowdLabels: The same annotations in sparse COO form.
    """
    pairs = [Y_cr[i, Y_mask[i]] for i in range(num_items)]
    return CrowdLabels.from_pairs(pairs, num_items=num_items)


def _read_annotator_config(cfg: dict, rng: np.random.Generator, default_behaviour: str) -> tuple:
    """Parses and repeats/shuffles the per-annotator precision, behaviour and region lists.

    Shared between the moons and circles loaders, which differ only in the
    default behaviour list.
    """
    A = cfg.get("n_annotators", 10)

    ann_prec = eval(cfg.get("annotator_prec", [0.8] * A))  # List of annotator precisions (probability of correct annotation)
    ann_prec = (ann_prec * (A // len(ann_prec) + 1))[:A]  # Repeat the list of precisions if it's shorter than A
    rng.shuffle(ann_prec)  # Shuffle the list of precisions to avoid any ordering effects

    ann_behaviour = eval(cfg.get("annotator_behaviour", [default_behaviour] * A))  # "spammer" or "adversarial" or "class_0" or "class_1" in certain regions.
    ann_behaviour = (ann_behaviour * (A // len(ann_behaviour) + 1))[:A]  # Repeat the list of behaviours if it's shorter than A
    rng.shuffle(ann_behaviour)  # Shuffle the list of behaviours to avoid any ordering effects

    ann_region = eval(cfg.get("annotator_region", "[(0, 0)]"), {"np": np})  # Cargar como lista de numpy
    ann_region = (ann_region * (A // len(ann_region) + 1))[:A]  # Repetir si faltan regiones para llegar a A anotadores.
    indices = rng.permutation(len(ann_region))
    ann_region = [ann_region[i] for i in indices]

    return A, ann_prec, ann_behaviour, ann_region


def load_config(path: str | Path) -> dict:
    """Loads a dataset config from a YAML file.

    Args:
        path: Path to a YAML file such as ``data/moons.yaml``. When the file
            has several commented-out configurations (as ``moons.yaml`` does),
            only the first active one is parsed.

    Returns:
        dict: The parsed config, as passed to [load_moons_dataset][gpcrowdkit_data.moons.load_moons_dataset].
    """
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def load_moons_dataset(cfg: dict, make_plots: bool = True) -> MoonsCrowdData:
    """Generates a crowdsourced "moons" dataset from a config dict.

    Args:
        cfg: Dataset configuration, in the format of ``data/moons.yaml``.
        make_plots: Whether to render the diagnostic plots (true labels, an
            annotator grid, and one plot per annotator) to ``cfg["dir_plots"]``.

    Returns:
        MoonsCrowdData: Training features, sparse crowd labels, and a clean
        held-out test split.
    """
    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)
    name = cfg.get("name", "moons")
    n_train = cfg.get("n_train", 2000)
    n_test = cfg.get("n_test", 500)
    noise_moons = cfg.get("noise_moons", 0.1)
    s = cfg.get("annotations_percentage", 0.2)  # Percentage of points annotated by each annotator
    dir_plots = cfg.get("dir_plots", f"results/{name}/plots/")

    A, ann_prec, ann_behaviour, ann_region = _read_annotator_config(cfg, rng, "spammer")

    X, Y = make_moons(n_train + n_test, noise=noise_moons, random_state=seed)

    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    Y_cr, Y_mask = _annotate(X_train, Y_train, ann_prec, ann_behaviour, ann_region, n_classes=2, S=s, rng=rng)
    labels = _to_crowd_labels(Y_cr, Y_mask, n_train)

    if make_plots:
        plot_true_labels(X, Y, title=f"True labels, noise={noise_moons} (Moons dataset)", save_path=dir_plots)
        plot_annotators_grid(
            X=X_train, Y_true=Y_train, Y_cr=Y_cr, Y_mask=Y_mask,
            ann_prec=ann_prec, ann_behaviour=ann_behaviour, ann_region=ann_region,
            save_path=dir_plots,
        )
        for a in range(A):
            plot_annotator(
                X=X_train, Y_true=Y_train, Y_cr=Y_cr, Y_mask=Y_mask,
                ann_prec=ann_prec, annotator_id=a, ann_behaviour=ann_behaviour,
                ann_region=ann_region, save_path=dir_plots,
            )

    return MoonsCrowdData(
        X_train=X_train,
        labels=labels,
        z_train=Y_train,
        X_test=X_test,
        z_test=Y_test,
        n_classes=2,
        n_annotators=A,
        class_names=["class_0", "class_1"],
    )


def load_circles_dataset(cfg: dict, make_plots: bool = True) -> MoonsCrowdData:
    """Generates a crowdsourced "circles" dataset from a config dict.

    Same annotation model and output format as [load_moons_dataset][gpcrowdkit_data.moons.load_moons_dataset], built
    over `sklearn.datasets.make_circles` instead of `make_moons`.

    Args:
        cfg: Dataset configuration, in the format of ``data/moons.yaml``.
        make_plots: Whether to render the diagnostic plots.

    Returns:
        MoonsCrowdData: Training features, sparse crowd labels, and a clean
        held-out test split.
    """
    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)
    name = cfg.get("name", "circles")
    n_train = cfg.get("n_train", 2000)
    n_test = cfg.get("n_test", 500)
    noise_circles = cfg.get("noise_circles", 0.1)
    s = cfg.get("annotations_percentage", 0.2)  # Percentage of points annotated by each annotator
    dir_plots = cfg.get("dir_plots", f"results/{name}/plots/")

    A, ann_prec, ann_behaviour, ann_region = _read_annotator_config(cfg, rng, "adversarial")

    X, Y = make_circles(n_train + n_test, noise=noise_circles, random_state=seed)

    idx = rng.permutation(len(X))  # Mezclar para el train-test split
    X, Y = X[idx], Y[idx]

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    Y_cr, Y_mask = _annotate(X_train, Y_train, ann_prec, ann_behaviour, ann_region, n_classes=2, S=s, rng=rng)
    labels = _to_crowd_labels(Y_cr, Y_mask, n_train)

    if make_plots:
        plot_true_labels(X_train, Y_train, title=f"True labels, noise={noise_circles} (Circles dataset)", save_path=dir_plots)
        plot_annotators_grid(
            X=X_train, Y_true=Y_train, Y_cr=Y_cr, Y_mask=Y_mask,
            ann_behaviour=ann_behaviour, ann_region=ann_region, ann_prec=ann_prec,
            save_path=dir_plots,
        )
        for a in range(A):
            plot_annotator(
                X=X_train, Y_true=Y_train, Y_cr=Y_cr, Y_mask=Y_mask,
                annotator_id=a, ann_behaviour=ann_behaviour, ann_region=ann_region,
                ann_prec=ann_prec, save_path=dir_plots,
            )

    return MoonsCrowdData(
        X_train=X_train,
        labels=labels,
        z_train=Y_train,
        X_test=X_test,
        z_test=Y_test,
        n_classes=2,
        n_annotators=A,
        class_names=["class_0", "class_1"],
    )

def angle_in_sector(theta: float, angle_min: float, angle_max: float) -> bool:
    """
    Comprueba si theta está dentro del sector angular [angle_min, angle_max].
    Resistente a valores fuera de [0, 2π] y a sectores que cruzan el 0.
    
    Args:
        theta:     Ángulo del punto (radianes), cualquier valor
        angle_min: Inicio del sector (radianes), cualquier valor
        angle_max: Fin del sector (radianes), cualquier valor
    """
    TWO_PI = 2 * math.pi

    if angle_min % TWO_PI == 0 and angle_max % TWO_PI == 0 and angle_min != angle_max:
        return True # El sector completo, cualquier ángulo es válido
    else:
        # Normalizar los tres ángulos a [0, 2π)
        t   = theta % TWO_PI
        lo  = angle_min % TWO_PI
        hi  = angle_max % TWO_PI

        if lo <= hi:
            # Caso normal: el sector NO cruza el 0
            # Ejemplo: [30°, 120°]
            return lo <= t <= hi
        else:
            # Caso cruce: el sector SÍ cruza el 0
            # Ejemplo: [330°, 30°]  →  t válido si t >= 330° OR t <= 30°
            return t >= lo or t <= hi


def _annotate(X, Y, ann_prec, ann_behaviour, ann_region, n_classes, S, rng):
    N = len(X)
    A = len(ann_prec)
    angles = np.arctan2(X[:, 1], X[:, 0]) # Compute the angle of each point in polar coordinates, used for behaviour-based annotation.
    annotation_mask = rng.binomial(1, S, size=(A, N)) # Binary mask indicating which annotator labels which data point
    counts = annotation_mask.sum(axis=0)
    M = counts.max()

    Y_cr = np.zeros((N, M, 2), dtype=int)   # Primera dim: data points;
                                            # Segunda dim: anotaciones para ese punto (hasta M como mucho);
                                            # tercera dim: [annot_id, label]
    Y_mask = np.zeros((N, M), dtype=bool)   # Indica qué anotadores anotan cada punto (True si el anotador anotó ese punto, False si no)

    for i in range(N):
        theta = angles[i]

        ann_ids = np.where(annotation_mask[:, i] == 1)[0] # Quien anota ese punto
        for pos, a in enumerate(ann_ids):
            prec = ann_prec[a]
            behaviour = ann_behaviour[a]
            region = ann_region[a]
            [low, high] = region
            Y_mask[i, pos] = True

            if behaviour == "spammer":
                if angle_in_sector(theta, low, high): # If in the "spammer" region
                    if rng.random() < 0.5: # Totally random annotation in the "spammer" region
                        y_hat = Y[i] # Correct annotation
                    else:
                        y_hat = (Y[i] +1) % 2 # Incorrect annotation
                    
                else:
                    if rng.random() < prec: 
                        y_hat = Y[i] # Correct annotation
                    else:
                        y_hat = (Y[i] + 1) % 2 # Incorrect annotation
                            
            elif behaviour == "adversarial":
                if angle_in_sector(theta, low, high): # If in the "adversarial" region
                    y_hat = (Y[i] +1) % 2 # Incorrect annotation                        
                else:
                    if rng.random() < prec: # Totally random annotation in the "adversarial" region

                        y_hat = Y[i] # Correct annotation
                    else:
                        y_hat = (Y[i] +1) % 2 # Incorrect annotationç
            
            elif behaviour == "class_0":
                if angle_in_sector(theta, low, high): # If in the "class_0" region
                    y_hat = 0 # Always zero annotation
                else: # Outside the "class_0" region, normal behaviour with annotator precision
                    if rng.random() < prec: 
                        y_hat = Y[i] # Correct annotation
                    else:
                        y_hat = (Y[i] +1) % 2 # Incorrect annotation

            elif behaviour == "class_1": 
                if angle_in_sector(theta, low, high): # If in the "class_1" region
                    y_hat = 1 # Always one annotation
                else: # Outside the "class_1" region, normal behaviour with annotator precision
                    if rng.random() < prec: 
                        y_hat = Y[i] # Correct annotation
                    else:
                        y_hat = (Y[i] +1) % 2 # Incorrect annotation
            else:
                raise ValueError(
                    f"Unknown annotator behaviour: {behaviour}"
                )
            
            Y_cr[i, pos, 0] = a
            Y_cr[i, pos, 1] = y_hat

            Y_mask[i, pos] = True
    return Y_cr, Y_mask

def extract_annotator_matrix(X, Y, ann_prec, ann_behaviour, ann_region, n_classes = 2):
    """
    Genera la probabilidadd de anotar el dato N como clase k por el anotador A, teniendo en cuenta la precisión del anotador y su comportamiento.

    Args:
        X: (N, 2) - Datos de entrada.
        Y: (N,) - Etiquetas verdaderas.
        ann_prec: List[float] - Precisión de cada anotador.
        ann_behaviour: List[str] - Comportamiento de cada anotador ("spammer", "adversarial", "class_0", "class_1").
        ann_region: List[Tuple[float, float]] - Región angular para cada anotador.
        n_classes: int - Número de clases.
        S: float - Porcentaje de puntos anotados por cada anotador.
         
    """ 
    angle = np.arctan2(X[:, 1], X[:, 0]) # Ángulo de cada punto en coordenadas polares, usado para la anotación basada en comportamiento.
    if angle_in_sector(angle, ann_region[0][0], ann_region[0][1]):
        # Si el punto está en la región especial del anotador
        if ann_behaviour == "spammer":
            # Anotador spammer: anota aleatoriamente
            prob = np.array([0.5, 0.5])  # Probabilidad uniforme para ambas clases
        elif ann_behaviour == "adversarial":
            # Anotador adversarial: anota la clase opuesta a la verdadera
            prob = np.zeros(n_classes)
            prob[(Y + 1) % n_classes] = 1.0  # Probabilidad 1 para la clase opuesta
        elif ann_behaviour == "class_0":
            # Anotador que siempre anota clase 0
            prob = np.zeros(n_classes)
            prob[0] = 1.0
        elif ann_behaviour == "class_1":
            # Anotador que siempre anota clase 1
            prob = np.zeros(n_classes)
            prob[1] = 1.0
        else:
            raise ValueError(f"Unknown annotator behaviour: {ann_behaviour}")
    else: 
        prob = np.zeros(n_classes)
        prob[Y] = ann_prec  # Probabilidad de anotar correctamente
        prob[(Y + 1) % n_classes] = 1 - ann_prec

    return prob
        


### Plots for the simulated datasets

def plot_annotators_grid(
    X,
    Y_true,
    Y_cr,
    Y_mask,
    ann_prec,
    ann_behaviour,
    ann_region,
    save_path,
    annotators=None,
):
    """
    Genera una figura 2x3 con varios anotadores.

    Verde: anotación correcta.
    Rojo: anotación incorrecta.

    Círculo: clase 0.
    Cruz: clase 1.
    """

    if annotators is None:
        annotators = range(min(6, len(ann_behaviour)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes = axes.flatten()

    xlim = (X[:, 0].min() - 0.2, X[:, 0].max() + 0.2)
    ylim = (X[:, 1].min() - 0.2, X[:, 1].max() + 0.2)

    radius = max(
        abs(xlim[0]),
        abs(xlim[1]),
        abs(ylim[0]),
        abs(ylim[1]),
    ) * 2

    for ax, annotator_id in zip(axes, annotators):

        behaviour = ann_behaviour[annotator_id]
        region = ann_region[annotator_id]
        [theta_min, theta_max] = region

        # --------------------------------------------------
        # Obtener ejemplos anotados por este anotador
        # --------------------------------------------------

        annotated = np.zeros(len(X), dtype=bool)
        correct = np.zeros(len(X), dtype=bool)

        for i in range(len(X)):

            rows = np.where(
                Y_mask[i]
                & (Y_cr[i, :, 0] == annotator_id)
            )[0]

            if len(rows) == 0:
                continue

            j = rows[0]

            y_hat = Y_cr[i, j, 1]

            annotated[i] = True
            correct[i] = (y_hat == Y_true[i])

        # --------------------------------------------------
        # Región sombreada
        # --------------------------------------------------

        wedge = Wedge(
            center=(0, 0),
            r=radius,
            theta1=np.degrees(theta_min),
            theta2=np.degrees(theta_max),
            alpha=0.15,
        )

        ax.add_patch(wedge)

        # --------------------------------------------------
        # Clase 0
        # --------------------------------------------------

        mask0_good = (
            (Y_true == 0)
            & annotated
            & correct
        )

        mask0_bad = (
            (Y_true == 0)
            & annotated
            & (~correct)
        )

        ax.scatter(
            X[mask0_good, 0],
            X[mask0_good, 1],
            marker="o",
            c="purple",
            s=25,
        )

        ax.scatter(
            X[mask0_bad, 0],
            X[mask0_bad, 1],
            marker="x",
            c="purple",
            s=25,
        )

        # --------------------------------------------------
        # Clase 1
        # --------------------------------------------------

        mask1_good = (
            (Y_true == 1)
            & annotated
            & correct
        )

        mask1_bad = (
            (Y_true == 1)
            & annotated
            & (~correct)
        )

        ax.scatter(
            X[mask1_good, 0],
            X[mask1_good, 1],
            marker="o",
            c="olive",
            s=25,
        )

        ax.scatter(
            X[mask1_bad, 0],
            X[mask1_bad, 1],
            marker="x",
            c="olive",
            s=25,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

        ax.set_title(
            f"Ann {annotator_id}\n{behaviour}\n Precision: {ann_prec[annotator_id]:.2f}"
        )

    # ocultar subplots sobrantes
    for ax in axes[len(list(annotators)):]:
        ax.axis("off")

    plt.tight_layout()
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='k',
            markerfacecolor='black',
            markersize=8,
            label='Correct annotation'
        ),
        Line2D(
            [0], [0],
            marker='x',
            color='k',
            markerfacecolor='black',
            markersize=8,
            label='Incorrect annotation'
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='olive',
            markersize=8,
            label='Class 1'
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='purple',
            markersize=8,
            label='Class 0'
        ),
        Patch(
            alpha=0.15,
            label='Special region'
        )
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = Path(save_path + 'annotators_grid.png')
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def plot_annotator(
    X,
    Y_true,
    Y_cr,
    Y_mask,
    annotator_id,
    ann_prec,
    ann_behaviour,
    ann_region,
    save_path,
):
    """
    Visualiza un único anotador.

    Clase 0 -> círculo
    Clase 1 -> cruz

    Verde -> anotación correcta
    Rojo -> anotación incorrecta

    Región sombreada -> comportamiento especial
    """

    behaviour = ann_behaviour[annotator_id]
    region = ann_region[annotator_id]
    [theta_min, theta_max] = region
    fig, ax = plt.subplots(figsize=(8, 8))

    # --------------------------------------------------
    # Encontrar ejemplos anotados por este anotador
    # --------------------------------------------------

    annotated = np.zeros(len(X), dtype=bool)
    correct = np.zeros(len(X), dtype=bool)

    for i in range(len(X)):

        rows = np.where(
            Y_mask[i]
            & (Y_cr[i, :, 0] == annotator_id)
        )[0]

        if len(rows) == 0:
            continue

        j = rows[0]

        y_hat = Y_cr[i, j, 1]

        annotated[i] = True
        correct[i] = (y_hat == Y_true[i])

    # --------------------------------------------------
    # Región sombreada
    # --------------------------------------------------

    xlim = (
        X[:, 0].min() - 0.2,
        X[:, 0].max() + 0.2,
    )

    ylim = (
        X[:, 1].min() - 0.2,
        X[:, 1].max() + 0.2,
    )

    radius = max(
        abs(xlim[0]),
        abs(xlim[1]),
        abs(ylim[0]),
        abs(ylim[1]),
    ) * 2

    wedge = Wedge(
        center=(0, 0),
        r=radius,
        theta1=np.degrees(theta_min),
        theta2=np.degrees(theta_max),
        alpha=0.15,
        label="Special region",
    )

    ax.add_patch(wedge)

    # --------------------------------------------------
    # Clase 0
    # --------------------------------------------------

    mask0_good = (
        (Y_true == 0)
        & annotated
        & correct
    )

    mask0_bad = (
        (Y_true == 0)
        & annotated
        & (~correct)
    )

    ax.scatter(
        X[mask0_good, 0],
        X[mask0_good, 1],
        marker="o",
        c="purple",
        s=35
    )

    ax.scatter(
        X[mask0_bad, 0],
        X[mask0_bad, 1],
        marker="x",
        c="purple",
        s=35
    )

    # --------------------------------------------------
    # Clase 1
    # --------------------------------------------------

    mask1_good = (
        (Y_true == 1)
        & annotated
        & correct
    )

    mask1_bad = (
        (Y_true == 1)
        & annotated
        & (~correct)
    )

    ax.scatter(
        X[mask1_good, 0],
        X[mask1_good, 1],
        marker="o",
        c="olive",
        s=35
    )

    ax.scatter(
        X[mask1_bad, 0],
        X[mask1_bad, 1],
        marker="x",
        c="olive",
        s=35
    )

    # --------------------------------------------------
    # No anotados
    # --------------------------------------------------

    not_annotated = ~annotated

    ax.scatter(
        X[not_annotated, 0],
        X[not_annotated, 1],
        marker=".",
        c="lightgray",
        alpha=0.3
    )

    # --------------------------------------------------

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    ax.set_title(
        f"Annotator {annotator_id} ({behaviour}, precision={ann_prec[annotator_id]:.2f})"
    )
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='purple',
            linestyle = 'None',
            markersize=8,
            label='Class 0'
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='olive',
            linestyle='None',
            markersize=8,
            label='Class 1'
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='gray',
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=8,
            label='Correct annotation'
        ),
        Line2D(
            [0], [0],
            marker='x',
            color='gray',
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=8,
            label='Incorrect annotation'
        ),

        Line2D(
            [0], [0],
            marker='.',
            color='lightgray',
            markersize=8,
            label='Not annotated'
        ),
    ]
    ax.legend(handles=legend_elements)

    save_path = Path(save_path + f'annotator_{annotator_id}_{ann_behaviour[annotator_id][0]}_in_{ann_region[annotator_id][0]:.2f}_{ann_region[annotator_id][1]:.2f}.png')

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def plot_true_labels(X, Y, title="True labels", save_path=None):
    """
    X: (N, 2)
    Y: (N,)
    """

    plt.figure(figsize=(7, 7))

    # Clase 0
    mask0 = (Y == 0)
    plt.scatter(
        X[mask0, 0],
        X[mask0, 1],
        c="tab:purple",
        marker="o",
        label="Class 0",
        alpha=0.7
    )

    # Clase 1
    mask1 = (Y == 1)
    plt.scatter(
        X[mask1, 0],
        X[mask1, 1],
        c="tab:olive",
        marker="x",
        label="Class 1",
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.2)

    save_path = Path(save_path + f'real_labels.png')

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()




    