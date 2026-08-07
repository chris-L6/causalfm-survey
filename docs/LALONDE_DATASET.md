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
  selection bias. This pairing (or a trimmed subset of it — see below) is
  the actual estimation task fed to every model in `Lalonde_benchmark.ipynb`
  as `(X, T, Y)`.

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

## The overlap problem, and trimming (`variant="nsw_psid_trimmed"`)

Even scored correctly, the untrimmed comparison has a second problem: NSW-treated
and PSID-controls barely overlap on covariates *at all* — verified directly:

| | NSW-treated | PSID-controls |
|---|---|---|
| age | 25.8 | 34.9 |
| married | 19% | 87% |
| nodegree | 71% | 31% |
| re74 (1974 earnings) | $2,096 | $19,429 |
| re75 (1975 earnings) | $1,532 | $19,063 |

This is severe enough that practically every method in `Lalonde_benchmark.ipynb`
gets the *sign* of the effect wrong on the untrimmed data, not just the
magnitude — not because the methods are bad, but because most PSID-control
units have no comparable treated unit to be compared against, so there's no
covariate-space region where "controlling for X" can actually work.

`load_lalonde(variant="nsw_psid_trimmed")` — what `Lalonde_benchmark.ipynb`
uses by default — restricts PSID-controls to **common propensity-score
support**: fit `p(treat=1|X)` on the pooled sample (`causal_bench/data_loader.py`,
`_trim_to_common_support`), keep only control units whose score falls
within the range actually observed among treated units. Verified effect:

```
n dropped: 1,392 of 2,490 PSID-control units (56%)
Naive observed diff, trimmed:   -$5,896.60   (vs. -$15,204.78 untrimmed)
```

Trimming doesn't eliminate confounding among the units that remain — the
naive diff-in-means is still far from the true $1,794 — but it removes the
impossible-by-construction part of the problem. Verified effect on actual
model accuracy (absolute ATE error, lower is better):

| Model | Untrimmed | Trimmed |
|---|---|---|
| S-learner | $2,561 | **$1,645** |
| Do-PFN | $7,010 | **$2,428** (now sign-correct: +$4,222) |
| CausalFM | $2,092 | $2,720 |
| Debiased ML | $12,278 | **$4,265** |
| T-learner / IPW | $15,262 | **$5,568** |
| DR | $16,205 | **$5,836** |
| X-learner | $11,305 | **$6,015** |

Every metalearner improves substantially, Do-PFN becomes sign-correct for
the first time, and S-learner becomes the most accurate model overall.
CausalFM is the one exception (slightly worse trimmed) — plausibly because
it's zero-shot on a smaller, differently-shaped context (896 fewer training
units) rather than being fit fresh to it like the metalearners are.

The untrimmed `variant="nsw_psid"` (the default if you don't pass `variant=`)
is still available if you want the original, maximally-hard comparison.

## Practical takeaway

If you extend this benchmark to other real-world datasets: check whether
your "ground truth" was actually measured independently (e.g., from a
randomized experiment, like here) or whether it's silently derived from the
same confounded data your models are being scored on. The latter isn't a
ground truth at all — it's the bias you're trying to measure.
