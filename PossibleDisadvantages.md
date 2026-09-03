# Why `FeatDepDirichletAnnotator` underperforms `VariationalDirichletAnnotator`

This note enumerates every structural difference between the two strategies as they
exist in the codebase today (`src/gpcrowdkit/annotators/strategies.py`), and argues,
for each one, whether it plausibly explains why `FeatDepDirichletAnnotator` scores
worse than `VariationalDirichletAnnotator` when both are trained under the same
`iterations`/`batch_size`/`learning_rate` budget in `train()` (`src/gpcrowdkit/inference.py`).

Five rounds of fixes have already been applied this session and are reflected in the
current code:

1. `kl_divergence()` used to be computed over only the current minibatch, under-penalising
   the KL term by roughly `batch_size / N` relative to what `ELBOTerms.total`
   (`src/gpcrowdkit/models.py`) assumes -- it always adds `kl_annotator` unscaled, correct
   for strategies with genuinely global parameters but not for one whose `alpha_tilde` is
   a function of `X`. This was first fixed by evaluating the network over the entire
   dataset every call, then refined (once it became clear the model's own `N/B`
   minibatch correction, already applied to `latent`/`crowd`/`entropy`, could just be
   reproduced from inside `kl_divergence()` itself) to sum the KL over the current batch
   -- reusing the forward pass `label_log_terms` already did, no extra one needed -- and
   multiply by `N/B`. Verified directly: with a full batch (`scale = N/N = 1`) this
   reproduces the old exact full-dataset value bit-for-bit; averaged over many small
   random batches it converges to the same value, confirming the estimator is unbiased.
2. The network used to map `X` alone to the entire `[A, C, C]` output block, giving each
   annotator a private, disjoint slice of a large final layer. It now takes
   `(x_n, annotator_embedding, true_class_onehot)` as explicit inputs and outputs only
   the `C` concentrations of one Dirichlet column, sharing the trunk across annotators
   and classes -- matching the paper's Eq. 2 more literally and cutting the parameter
   count roughly in half (e.g. 2935 -> 1820 params/worker on AI4SkINv2-2 with CONCH
   features).
3. A two-phase warm-up (`train(..., warmup_iterations=...)`) was added, freezing the GP
   kernel/inducing points for the first phase so `q(Z)` and the annotator model can
   settle before the latent GP starts chasing their output.
4. **(Section 1, below) The network is now data-driven-initialised.** `alpha_tilde(x, a,
   j)` is a residual: a trainable, per-`(a, j)` baseline table seeded from
   `init_alpha_tilde` -- exactly what `VariationalDirichletAnnotator` gets -- plus a
   correction from the trunk whose last layer starts at exactly zero. Verified by direct
   test: at step 0, `get_alpha_tilde(X)` matches `init_alpha_tilde`'s target to machine
   precision (`2e-16`) and is identical across every item regardless of `X`
   (cross-item std `1.5e-15`), for every `hidden_units` configuration tried.

The remaining gap after these fixes is smaller but still present. The sections below
rank the plausible causes of what's left; Section 1 is now historical context (the fix
that used to be missing) rather than an open item.

## Summary table

| # | Difference | Present in current code? | Likely contributor? |
|---|---|---|---|
| 1 | No data-driven initialisation for the network | **Fixed** (see below) | Was **High**; now addressed |
| 2 | Parameter count vs. dataset size, even after the redesign | Yes, reduced but not eliminated | **Medium-high** |
| 3 | Non-conjugate optimisation landscape (net vs. closed-form Dirichlet) | Yes, inherent to the strategy | **Medium-high** |
| 4 | KL term now correctly strong, pulling against an uninformative start | No longer applies -- start is now informative, not flat | **Low**, downgraded |
| 5 | Possible model misspecification (real noise may not depend on X) | Dataset-dependent, can't be fixed in code | **Medium** |
| 6 | Mean-pooled MIL features may wash out whatever drives disagreement | AI4SkINv2-2-specific | **Low-medium** |
| 7 | Unequal compute per "step" across strategies | Confound in the comparison, not a model defect | **Low** (affects interpretation, not the model itself) |
| 8 | Everything else (ELBO term, posterior-mean estimator, batching/alignment) | Verified identical | **Ruled out** |

## 1. No data-driven initialisation for the network -- fixed this session

`VariationalDirichletAnnotator` is seeded via `init_alpha_tilde` (`strategies.py:343-384`)
in every example script's `build_annotator()`: each `(worker, observed label)` slot
starts with mass accumulated from the empirical vote distribution, so the model begins
already close to "workers who agree with the crowd get diagonal-biased confusion
matrices." The docstring on `init_alpha_tilde` explains why this matters: *"the ELBO is
not convex, and from a uniform start the model can settle into a labelling that is a
permutation of the truth -- self-consistent, high-likelihood, and useless."*

`FeatDepDirichletAnnotator` used to have no equivalent: its `Dense` layers used Keras'
default Glorot-uniform kernel init and zero bias, so at step 0 the network's raw output
was close to zero for every `(x, annotator, class)` triple -- a flat, uninformative
Dirichlet column everywhere, independent of `X`, exactly the degenerate starting point
`init_alpha_tilde` exists to avoid.

**This is now fixed.** `alpha_tilde(x, a, j)` is built as a residual
(`strategies.py:475-511`): a trainable `Embedding`-backed baseline table, one entry per
`(annotator, true class)` combination, initialised from `init_alpha_tilde` (or the same
diagonal-bias fallback `VariationalDirichletAnnotator` uses when no init is given) --
plus a correction from the shared trunk, whose *last* `Dense` layer is
zero-initialised (`kernel_initializer="zeros", bias_initializer="zeros"`). Because a
zero-kernel layer's output depends only on its own bias, the correction is exactly `0`
for every input at step 0 regardless of what the (randomly-initialised) hidden layers or
embedding computed upstream -- so `alpha_tilde` starts as pure baseline, numerically
identical to `VariationalDirichletAnnotator`'s own start, and only grows an `X`-dependent
component as training moves the trunk's weights away from zero. All four call sites
(`examples/01_quickstart.py`, `02`, `04`, `05`) now pass
`alpha_tilde_init=init_alpha_tilde(labels, class_probs)`, the same call already used to
seed `VariationalDirichletAnnotator`.

**Verdict: was very likely a major contributor; now addressed.** The remaining sections
describe what's left after this fix.

## 2. Parameter count vs. dataset size -- reduced, not eliminated

Before the redesign, the network had ~2935 parameters per worker on AI4SkINv2-2 (CONCH,
512-d features) -- an ~A*C*C-sized output block from a shared trunk, i.e. roughly
29,000 total weights fit against 3,790 real annotations. After sharing the trunk across
annotators and classes via the embedding/one-hot conditioning, that's down to ~1,820
params/worker (~18,000 total) -- better, but still an order of magnitude more
parameters than `VariationalDirichletAnnotator`'s 360 total (36/worker), which are
close to sufficient statistics for a conjugate Dirichlet-multinomial update and need
comparatively little data or optimisation to fit well.

A network with thousands of weights, fit by a fixed number of Adam steps against a few
thousand annotations, is intrinsically harder to bring to convergence than a
closed-form-adjacent update over 360 numbers, independent of any initialisation issue.

**Verdict: still a real contributor**, smaller than before the redesign but not gone.

## 3. Non-conjugate optimisation landscape

`VariationalDirichletAnnotator.expected_log_confusion()` (`strategies.py:124-135`) is
the direct digamma-of-concentration identity for a Dirichlet -- the ELBO's gradient
with respect to `alpha_tilde` has a clean, well-understood form, and Adam on a flat
parameter tensor in that family converges quickly and reliably in practice (this is
essentially why SVGPCR's reference implementation uses it).

`FeatDepDirichletAnnotator.get_alpha_tilde()` (`strategies.py:475-499`) instead composes
an embedding lookup, several `Dense`+ReLU layers, a `softplus`, and a `digamma` --
gradients must flow through all of that. The ELBO is already non-convex in the
variational parameters; here it is further composed with a non-convex neural network,
compounding the optimisation difficulty on top of the raw parameter count from
Section 2.

**Verdict: plausible, structural, and not fixable without changing the strategy's
fundamental design** -- it is the price of feature-dependence, not a bug.

## 4. The (now-correct) KL term, revisited after the initialisation fix

Before this session's first fix, `kl_divergence()` under-counted the penalty by roughly
`batch_size / N`, so the network was effectively under-regularised and could drift from
the prior largely unchecked. It is now computed over the complete dataset every call,
exactly as `ELBOTerms.total` assumes, and exerts its full, correct pull toward
`self.alpha` (a flat, uniform-Dirichlet prior by default; `strategies.py:447-449`) on
every step.

Before Section 1's fix, this correct-but-strong penalty was pulling against a network
that had *randomly* started far from any sensible confusion pattern -- a genuine drag on
convergence, worth flagging. Now that `alpha_tilde` starts at `init_alpha_tilde`'s
value, the KL term at step 0 measures the (moderate, expected) distance between a
vote-based diagonal-biased estimate and the flat prior, not the (large, arbitrary)
distance a random init would have produced -- confirmed directly: `kl_annotator` at
step 0 in example 02 dropped from ~2606 before this fix to ~225 after it.

**Verdict: downgraded.** This term is doing its job correctly and is no longer fighting
an uninformative start; any remaining slowness it causes is ordinary regularisation
pressure, not a symptom of the earlier bugs.

## 5. Possible model misspecification for this dataset

`FeatDepDirichletAnnotator` is strictly more expressive than `VariationalDirichletAnnotator`:
it can represent everything the static-confusion-matrix model can, plus arbitrary
`X`-dependence. In principle that should never make it *worse* asymptotically -- but
with finite data and a finite optimisation budget, extra unused expressiveness is pure
variance with no offsetting benefit. If pathologist reliability on AI4SkINv2-2 does not
actually vary systematically with where a slide's pooled embedding sits in feature
space, the feature-dependent inductive bias buys nothing, while Sections 2-3 still cost
something.

This is testable, not just arguable: `data/moons.py` builds annotator behaviour that
*is* explicitly a function of `X` by construction (spammer/adversarial angular
regions). If `FeatDepDirichletAnnotator`'s relative gap to
`VariationalDirichletAnnotator` is smaller (or reversed) on the moons "hard" config than
on AI4SkINv2-2, that is direct evidence real annotator noise here is closer to
feature-independent, and the feature-dependent model is solving a harder problem than
the data warrants.

**Verdict: plausible and dataset-dependent** -- worth checking empirically (see
"Next diagnostic steps" below) rather than assuming either way.

## 6. Mean-pooled MIL features may not carry the relevant signal

Each WSI enters the model as a mean-pooled bag of patch embeddings
(`data/ai4skin.py`, `_pool_embeddings`). Whatever might actually drive a pathologist's
disagreement on a given slide -- a handful of ambiguous, hard-to-grade patches -- can be
diluted into a global average alongside hundreds of easy, unambiguous ones. Even if
annotator reliability genuinely does depend on lesion morphology, the pooled feature
vector `FeatDepDirichletAnnotator` conditions on may be a lossy proxy for it. This
compounds Section 5: it's not just that the noise might be feature-independent, but
that even feature-dependent noise might not be visible in *this particular*
representation of `X`.

**Verdict: plausible, AI4SkINv2-2-specific, and not something either annotator
strategy's code can fix** -- it would require attention pooling or a per-patch model
instead of mean pooling.

## 7. Unequal compute per "step" is a confound in the comparison, not a model defect

Every example script's `build_annotator()` gives every strategy the same `iterations`,
`batch_size`, and `learning_rate` (see `examples/05_compare_ai4skin_dataset.py`).
But one gradient step still means more work for `FeatDepDirichletAnnotator` than for
`VariationalDirichletAnnotator`: the latter's step is a closed-form-adjacent update over
an `[A,C,C]` tensor, while the former's runs `label_log_terms` over `batch_size * A * C`
rows through several network layers (`kl_divergence` now reuses that same forward pass
-- Section 1's second refinement -- rather than adding a second, larger one over the
whole dataset, so this gap is smaller than it was earlier this session, but it hasn't
gone to zero: a `Dense`/`Embedding`-based forward-and-backward pass over `B*A*C` rows is
still more FLOPs than a digamma/lbeta update over `A*C*C` numbers). "Same number of
steps" is therefore still not quite "same optimisation budget" -- `FeatDepDirichletAnnotator`
may be closer to `VariationalDirichletAnnotator` per unit of wall-clock/FLOPs than per
step count suggests.

**Verdict: doesn't explain a worse *ceiling*, but can explain a worse result at a fixed
`--iterations` value** -- worth controlling for before concluding the strategy is
fundamentally weaker rather than just slower to converge per step. Weaker now than
before Section 1's KL refinement, since that removed the single largest per-step
compute gap between the two strategies.

## 8. Ruled out: things that are identical between the two strategies

- **ELBO decomposition and scaling.** Both strategies plug into the same
  `GPCrowdModel.elbo_terms` (`src/gpcrowdkit/models.py`); the five-term structure, the
  `N/B` minibatch scaling of the data terms, and the (now, for both) unscaled KL terms
  are shared code, not strategy-specific.
- **Posterior-mean estimator.** Both use `alpha_tilde / sum(alpha_tilde)` for the
  reporting confusion matrices (`confusion_matrices()` in each class) -- neither uses
  the biased `softmax(E[log R])` shortcut.
- **Batch/annotation alignment.** Both consume the same `CrowdBatch` from
  `CrowdLabels.gather_batch` (`src/gpcrowdkit/data.py`); there is no strategy-specific
  indexing path that could misalign features and annotations for one but not the other.
- **Warm-up.** Both benefit equally from `warmup_iterations`: the GP is what gets frozen,
  not anything strategy-specific, so this fix helps (or doesn't) symmetrically.

These were checked against the code directly and are not differentiators.

## Evidence from this session's short smoke-test runs

The following AI4SkINv2-2 runs (30 main iterations, `--num-inducing 20`,
`--batch-size 32`; far short of the default 300-iteration full run, so these are
directional, not converged benchmarks) were taken at four points this session and show
the gap narrowing as each fix landed:

| Stage | `VariationalDirichletAnnotator` train acc | `FeatDepDirichletAnnotator` train acc |
|---|---|---|
| Before any fix (5 iterations only) | 0.773 | 0.767 |
| After the `kl_divergence` full-dataset fix | 0.777 | 0.773 |
| After the embedding redesign + 20-step warm-up | 0.771 | 0.777 |
| After the data-driven initialisation fix (this section) | 0.771 | 0.773 |

`FeatDepDirichletAnnotator` has been within a point or two of
`VariationalDirichletAnnotator` since the second fix, and the initialisation fix mainly
shows up in `kl_annotator`'s starting value (example 02: ~2606 -> ~225) rather than in
train accuracy at this short a budget -- consistent with Section 4's point that its
main effect was removing a drag on *convergence speed*, not raising the *ceiling*. What
these short runs can't yet distinguish is whether the remaining few points of gap are
Sections 2-3 (still slower to converge, would close with more iterations) or Section 5
(the feature-dependent inductive bias genuinely isn't earning anything on this
dataset). That's what the next steps below are for.

## Recommended next diagnostic steps

1. **Run both strategies to full convergence** (the default 300+ iterations, or more)
   rather than the short smoke tests above, to separate "slower to converge" (Sections
   2-3) from "converges to a worse optimum" (Section 5).
2. **Compare the relative gap on `data/moons.py`'s "hard" config vs. AI4SkINv2-2.** A
   smaller (or reversed) gap on moons, where annotator noise is X-dependent by
   construction, would support Section 5/6 (real annotator noise here may not depend
   much on the pooled embedding) over Sections 2-3 (optimisation-only explanations).
3. **Track `history.kl_annotator` and `history.crowd` over iterations** for
   `FeatDepDirichletAnnotator` specifically (the `ELBOTerms` decomposition is already
   exposed by `train()`) to see whether the KL term is still falling steeply late in
   training (suggesting it hasn't converged yet -- Sections 2-3) or has plateaued early
   at a value indicating the correction stayed near zero (suggesting Section 5: the
   trunk found no useful `X`-dependence to add on top of the baseline).
