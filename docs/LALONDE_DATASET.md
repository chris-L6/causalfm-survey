# The Lalonde Dataset: What It Is, and Why We Fixed How We Score It

## What it is

The Lalonde dataset comes from Robert LaLonde's 1986 study of the National
Supported Work (NSW) program, a job-training intervention. Dehejia & Wahba
(1999, 2002) re-released the data in the form this repo uses, hosted at
Dehejia's NBER page. It's a standard benchmark in causal inference precisely
because it contains **two different ways to estimate the same treatment
effect**, one trustworthy and one not:

- **NSW-treated vs. NSW-control** — a genuine **randomized experiment**.
  Because assignment to treatment was random, a simple difference in mean
  `re78` (1978 earnings) between these two groups is an *unbiased* estimate
  of the true average treatment effect. No adjustment needed.
- **NSW-treated vs. PSID-controls** — an **observational** substitute.
  LaLonde's point (and the reason this dataset is famous) was to ask: if you
  only had non-experimental controls, could standard econometric methods
  still recover the right answer? PSID controls are a very different
  population from NSW participants (older, more work history, higher
  earnings generally), so a naive comparison is badly confounded by
  selection bias. This pairing is the actual estimation task — it's what
  gets passed to every model in `Lalonde_benchmark.ipynb` as `(X, T, Y)`.

Features: `age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `re74`,
`re75`. Treatment: `treat`. Outcome: `re78`.

## The numbers, verified directly

```
True experimental ATE   (NSW-treated vs. NSW-control):  $1,794.34
Naive observed diff     (NSW-treated vs. PSID-controls): -$15,204.78
```

The true effect is a modest positive number — the job-training program
helped, a bit. The naive comparison says the opposite and by a huge margin,
purely because PSID controls earn far more than NSW participants for reasons
that have nothing to do with the program. That gap **is** the selection
bias this dataset exists to test whether a causal method can correct for.

## What was wrong before, and the fix

`causal_bench/data_loader.py` originally computed `ds.ate` as the naive
diff-in-means on the *same* NSW-treated/PSID-controls data fed to every
model — i.e., it scored models against the confounded number, not the true
one. That's backwards: a method that stayed close to `-$15,204` wasn't
doing well, it was failing to adjust for confounding at all, and a method
that moved far away from it — toward the true small positive effect — was
being penalized for doing its job correctly.

`load_lalonde()` now loads the NSW-treated vs. NSW-control randomized
comparison separately and uses *that* diff-in-means as `ds.ate` — the true
benchmark to score against. The old naive number is preserved as
`ds.ate_naive_observed`, printed in the notebook for context, so the scale
of the selection-bias problem stays visible.

**Individual-level CATE still has no ground truth on this dataset** — only
the population-level ATE is checkable this way. That was true before and
remains true.

## Why this flipped the benchmark results

Scored against the corrected true ATE, DR and Debiased ML — which had
looked best under the old (wrong) scoring, because they landed close to the
naive `-$15,204` — turned out to be the *least* accurate models in the
table. They weren't correcting for the selection bias; they were mostly
reproducing it. CausalFM, previously worst, became the most accurate model
once scored correctly (and once its inputs were standardized — see
CLAUDE.md's "Models & dependencies" caveats for that separate fix). This
matches the point LaLonde's original paper was making: this exact
comparison is hard, and many standard adjustment methods fail to recover
the true effect from it.

## Practical takeaway

If you extend this benchmark to other real-world datasets: check whether
your "ground truth" was actually measured independently (e.g., from a
randomized experiment, like here) or whether it's silently derived from the
same confounded data your models are being scored on. The latter isn't a
ground truth at all — it's the bias you're trying to measure.
