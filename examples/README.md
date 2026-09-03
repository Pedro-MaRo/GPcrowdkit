# Examples

Runnable sanity checks for the library, in increasing order of detail. Each is
a plain script, not a notebook, so it can be run in CI or from the command
line as a regression check on the modelling code.

| Script | What it checks |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | End-to-end fit; asserts the model beats majority vote. |
| [`02_compare_annotator_strategies.py`](02_compare_annotator_strategies.py) | The three shipped `AnnotatorModel` strategies are interchangeable behind the same `GPCrowdModel`. |
| [`03_diagnostics_plots.py`](03_diagnostics_plots.py) | Plots the ELBO decomposition and true-vs-recovered confusion matrices to `output/`. Requires `pip install -e ".[examples]"`. |
| [`04_compare_moons_dataset.py`](04_compare_moons_dataset.py) | All four annotator strategies vs. majority vote on the synthetic "moons" dataset (`data/moons.py`), scored both on the annotated training items and on a held-out test split with no annotations. Requires `pip install -e ".[data]"`. |
| [`05_compare_ai4skin_dataset.py`](05_compare_ai4skin_dataset.py) | The same comparison on AI4SkINv2-2 (`data/ai4skin.py`), a real crowdsourced dataset of pathologist-graded skin whole-slide images, also scored against the dataset's own published Dawid-Skene/GLAD/MACE aggregators. Requires `pip install -e ".[data]"` and a checkout of AI4SkINv2-2. |

Run any of them with the project's virtualenv and GPU library paths already
configured:

```bash
./run.sh examples/01_quickstart.py
```

## Iteration counts

Every example trains with `train(..., early_stopping=True)`: each strategy
stops on its own once its ELBO plateaus, rather than all strategies running
for the same fixed number of steps -- a one-parameter-per-worker strategy
like `OneCoinAnnotator` typically converges, and stops, far sooner than a
high-capacity one like `FeatDepDirichletAnnotator`. `--iterations` is only the
upper bound in that case; the number of steps actually run is reported as
`iterations` in each script's summary. Pass `--no-early-stopping` to always
train for the full `--iterations` instead.
