# Kalman Filter Simulation Toolkit

![code size](https://img.shields.io/github/languages/code-size/ivaquero/blog-filters.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/blog-filters.svg)

A beginner-friendly tutorial on Kalman-family filters, reorganized from
Richard Labbe's
[Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
into a Jupyter Book plus a small Python package (`filters`, `models`, `plots`,
`simulators`, `symbols`).

## Read the book online

<https://ivaquero.github.io/blog-filters/>

## Run locally

The toolkit targets **Python 3.14** and is managed with [uv](https://docs.astral.sh/uv/):

```bash
# 1. Create the pinned environment (book tooling lives in the `book` extra)
uv sync --all-extras         # fallback: pip install -e ".[book,dev,sim]"

# 2. Build the book with Jupyter Book 2 (MyST). `--html` renders the static
#    site, `--execute` runs every notebook; outputs are cached in _build/execute.
uv run jupyter-book build --html --execute

# 3. Open the rendered HTML
open _build/html/index.html      # Windows: start _build/html/index.html
```

Continuous integration mirrors the local workflow — `ruff check`, `ruff format
--check`, `ty check --error-on-warning`, `pytest`, and finally
`jupyter-book build --html --execute` (with `BASE_URL=/blog-filters`, which is
how GitHub project pages are served) — and runs on every push.

## Chapter ↔ module map

| Chapter (Part 1 · Filter Implementation) | Filter modules | Support modules |
|---|---|---|
| `filters-bayes` — Grid Bayes filter | `filters.bayes` | `plots.plot_bayes` |
| `filters-kf-basic` — Linear Kalman filter | `filters.bayes`, `filters.stats` | `models.const_vel`, `plots.plot_bayes`, `plots.plot_common`, `plots.plot_kf`, `simulators.datagen`, `simulators.linear` |
| `filters-kf-design` — Designing the KF (Q, R, tuning) | `filters.kalman`, `filters.helpers`, `filters.stats` | `models.const_acc`, `models.const_vel`, `models.noise`, `plots.plot_common`, `plots.plot_kf`, `simulators.linear` |
| `filters-ghk` — α-β and α-β-γ filters | `filters.ghk` | `plots.plot_gh`, `simulators.datagen` |
| `filters-kf-plus` — Nonlinear KF (UKF, EnKF) | `filters.kalman`, `filters.kalman_ukf`, `filters.kalman_enkf`, `filters.sigma_points` | `models.noise`, `plots.plot_common`, `plots.plot_kf_plus` |
| `filters-smoothers` — RTS & fixed-lag smoothers | `filters.kalman`, `filters.smoothers` | `plots.plot_smoother` |
| `filters-pf` — Particle filter | (inlined in the notebook) | `plots.plot_pf`, `simulators.robot` |

| Chapter (Part 2 · Real Scenario Simulation) | Filter modules | Support modules |
|---|---|---|
| `filters-maneuver` — IMM maneuvering target | `filters.imm`, `filters.kalman` | `models.const_acc`, `models.const_vel`, `models.noise`, `plots.plot_common`, `plots.plot_kf`, `simulators.maneuver` |
| `filters-task-tracking` — Radar / robot / projectile tracking | `filters.kalman`, `filters.kalman_ekf`, `filters.kalman_ukf`, `filters.sigma_points` | `models.const_acc`, `models.const_vel`, `models.noise`, `plots.plot_common`, `simulators.radar`, `simulators.robot`, `simulators.trajectory` |
| `filters-task-fusion` — Two-sensor data fusion | `filters.fusion`, `filters.kalman` | `models.const_acc`, `models.const_vel`, `models.noise`, `plots.plot_common`, `plots.plot_kf`, `simulators.datagen`, `simulators.linear`, `simulators.radar` |

## Toolkit structure

### `filters` — filter implementations

- [x] `bayes` — grid Bayesian filter
- [x] `fusion` — multi-sensor fusion helpers
- [x] `ghk` — α-β-γ filtering
- [ ] `ghq` — Gauss-Hermite quadrature helpers (not yet integrated into a chapter)
- [x] `helpers` — `KFSaver` and other auxiliary tools
- [x] `imm` — interacting multiple models
- [x] `kalman` — linear Kalman filter
- [ ] `kalman_ckf` — cubature Kalman filter (not yet integrated into a chapter)
- [x] `kalman_ekf` — extended Kalman filter
- [x] `kalman_enkf` — ensemble Kalman filter
- [ ] `kalman_fm` — fading-memory filter (not yet integrated into a chapter)
- [ ] `kalman_hinf` — H∞ filter (not yet integrated into a chapter)
- [x] `kalman_ukf` — unscented Kalman filter
- [ ] `lsq` — least-squares filter (not yet integrated into a chapter)
- [ ] `particle` — particle filter class (not yet integrated; the PF chapter inlines its own loop)
- [ ] `resamplers` — particle resampling schemes (not yet integrated into a chapter)
- [x] `sigma_points` — unscented sigma-point generators
- [x] `smoothers` — RTS and fixed-lag smoothers
- [ ] `solvers` — Runge-Kutta and ODE helpers (not yet integrated into a chapter)
- [x] `stats` — statistical helpers (`gaussian`, `gaussian_pdf`, …)
- [ ] `transformers` — measurement / state transforms (not yet integrated into a chapter)

### `models` — motion & noise models

- [x] `const_acc` — constant-acceleration model
- [x] `const_vel` — constant-velocity model
- [x] `coord_turn` — coordinated-turn model
- [x] `clutter` — Poisson clutter generator (used by scenario code, not by any chapter yet)
- [x] `noise` — process & measurement noise helpers
- [x] `ssmodel` / `ssmodel_linear` / `ssmodel_nonlinear` — state-space base classes

### `plots` — plotting helpers

- [x] `plot_bayes` — Bayesian grid plots
- [x] `plot_common` — measurement / trajectory / residual plots
- [x] `plot_gh` — α-β-γ plots
- [x] `plot_kf` — linear KF plots
- [x] `plot_kf_plus` — nonlinear KF plots (UKF / EnKF / sigma points)
- [x] `plot_pf` — particle-filter plots
- [x] `plot_smoother` — smoother plots

### `simulators` — synthetic data & scenarios

- [x] `datagen` — CV/CA trajectory & measurement generators
- [x] `linear` — linear-motion simulator
- [x] `maneuver` — maneuvering-target simulator
- [x] `radar` — ground-radar simulator
- [x] `robot` — differential-drive robot simulator
- [x] `trajectory` — projectile / ballistic simulator

### `symbols` — symbolic derivations

- [x] `datagen` — SymPy derivations of the discrete-time Q matrix used in the KF-design chapter

## Notes on the code

The toolkit is a clean-room reorganisation of the filterpy codebase rather than
a compatibility fork: the goal is to demonstrate what a Kalman-family teaching
library looks like when it is written **for** Python 3.14 instead of ported to
it. That shows up in three places.

* **Typing is first-class.** Every package ships a PEP 561 `py.typed` marker
  and uses PEP 695 `type` aliases plus `numpy.typing.NDArray[np.float64]` in
  `filters._typing`. `KalmanFilter`, `ExtendedKalmanFilter`, and friends
  satisfy the runtime-checkable `Filter` protocol, so IMM / MMAE / plotting
  helpers accept anything structural instead of requiring inheritance.
* **Batch and smoother APIs return `NamedTuple`s.** `BatchOutput`,
  `Batch2Output`, `RtsOutput`, `UkfRtsOutput`, `SmootherOutput`, and
  `IMMOutput` expose `.means` / `.covs` / `.x` / `.P` fields while still
  unpacking as ordinary tuples — no existing notebook or test breaks.
* **Randomness is injected, not global.** Simulators, plotting helpers,
  resamplers, and clutter models take an explicit
  `numpy.random.Generator` (`rng=`) resolved through `filters._typing.resolve_rng`.
  The legacy `seed=` keyword is retired; pass `rng=np.random.default_rng(seed)`
  at the call site to pin reproducibility bit-for-bit.

Every module is a plain Python package (no `sys.path` hacks) so you can
`uv sync` or `pip install -e .` and import from anywhere.

## TODO

- Chapter-end **Summary / Key terms / Exercises / Further reading** sections
  (present in the original rlabbe book, not yet ported).
- Wire the "not yet integrated" filters (`ghq`, `kalman_ckf`, `kalman_fm`,
  `kalman_hinf`, `lsq`, `particle`, `resamplers`, `solvers`, `transformers`)
  into dedicated chapters.
- Add PDA / clutter-aware tracking chapter that consumes `models.clutter`.
- Add Singer motion model (`models.singer`) for maneuver-target literature.

## License

Content derived from
[Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
by Richard Labbe, distributed under the MIT license.
