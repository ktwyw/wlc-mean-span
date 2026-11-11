# Tutorial: walk-through and verification of the article

Ten executed Jupyter notebooks (an overview and nine chapters) that follow the article topic by topic. Each states a result,
derives it in more detail than the article can (symbolically with SymPy where possible), verifies it
against the archived data or by a live computation with the same code that produced the data, and
records its status: exact, derived or conditional.

| Notebook | Topic | Live computations |
|---|---|---|
| `00_overview.ipynb` | map of the article | — |
| `01_model_and_span.ipynb` | Model and definitions | 4000 chains sampled with the heat-kernel tangent law; span vs reference |
| `02_first_return_identity.ipynb` | The first-return identity | both derivations step by step; growth of the span at the ends of equilibrium chains ; return lengths of flux-weighted starts, the first-return identity and ∂X/∂p; concavity and F ≤ λ/2 on the table; Hessian in SymPy |
| `03_radius_of_gyration.ipynb` | Span, radius of gyration and depletion | g(λ) by symbolic integration; 7/1920; long-chain ratio coefficients; the identity δ = X/2 step by step and on sampled chains; the depletion wall offset |
| `04_short_chains.ipynb` | Short chains | J = 37/192 by quadrature and two symbolic routes; each step of the reduction of J; the limit function Q_ret and the dominating envelope; 9/256; dimension independence |
| `05_long_chains.ipynb` | Long chains | Var Q; the matched expansion to second order in SymPy; the Milne problem solved live at orders 16–256 (z_M, M₁, M₂, B); the moment identities; the reciprocity value of a₂ against a direct solution of the second-order problem; ψ_κ and s₄ = −11/135; the telegraph expansion |
| `06_spectral_solver.ipynb` | Numerical solution | the solver built step by step; the transfer formula rederived mode by mode; the alias series and its bound shown on the two-mode problem; Schur vs eigendecomposition; conditioning audit; alias-corrected inversion; one reference node recomputed and matched to 10⁻⁹; m⁻³ convergence and the high-order checks; inversion-noise audit; exact two-mode test |
| `07_independent_checks.ipynb` | Independent tests of the solution | finite-volume solutions on three grids; observed convergence orders; archived path comparison, including L/p = 10³ and 10⁴; Fig. S1 |
| `08_practical_formula.ipynb` | The practical formula | constraints of the practical formula in exact arithmetic; sign audit; error on fitted and withheld nodes; Fig. 4 |
| `09_verification_summary.ipynb` | all | the article's quantitative statements recomputed: 117 checks (values to their printed rounding, bounds one-sidedly) |

## Where each result is derived or checked

Results are listed by name, in the order in which the article presents them. The middle column gives the
equation number in the submitted article, as a guide for readers who have it; the numbers may change in the
published version, and the names will not.

| Result | Equation in the article | Notebook and section |
|---|---|---|
| Model, span and radius of gyration (definitions) | (1)–(3) | 01, "The model"; 03, "Derivation" |
| Growth of the span at a record | (5) | 02, "The pathwise derivation", steps 1–2, and the first test |
| First-return identity | (6) | 02, "Derivation 1", "Derivation 2", steps 3–5, and the second test |
| Concavity, F ≤ λ/2, ∂X/∂p, joint concavity | (7) | 02, "Consequences checked on the reference table", "Transform and derivatives" |
| Kinetic equation and its flux form | (8), (9) | 02, "Derivation 1"; 06, "The problem to solve" |
| Grazing-layer scaling | (10) | 04, "Why the first correction is of order λ²", "The limit function Q_ret" |
| Return kernel and J = 37/192 | (11), (12) | 04, "The return kernel", "The reduction of J to a rational integral, step by step" |
| Short-chain expansion, 37/768 | (13) | 04, "From J to the WLC coefficient", "Numerical evidence" |
| Endpoint excess, 9/256 | (14) | 04, "The endpoint excess" |
| Leading Gaussian law | (15) | 05, "The leading Gaussian law" |
| Milne problem and first-order matching | (16), (17) | 05, "From the boundary layer to the offset", steps 1–6; "The kinetic boundary layer" |
| Long-chain expansion with B | (18) | 05, steps 5–6; 06, "Convergence in angular order" (comparison with the table) |
| Moment representation of B | (19) | 05, "The moment representation of B", "The constant a₂ by reciprocity" |
| Radius of gyration and the ratio X/R_g | (20)–(23) | 03, "Derivation", "The ratio and its limits" |
| Depletion identity δ = X/2 | (24) | 03, "The identity δ = X/2, step by step" |
| Wall offset of the depletion layer | (25) | 03, "The depletion layer at a wall" |
| Practical formula | (26) | 08 |
| Telegraph process | (28) | 05, "The telegraph process"; 06, "An exactly solvable test of the inversion" |
| Nodal system and transfer formula |  | 06, steps 2–3, "Step 3 in formulas" |
| Fourier-series inversion and alias correction |  | 06, step 4, "Where the alias series comes from" |
| Generality of the identity |  | 02, "Derivation 2" |
| Other velocity processes, dimension independence |  | 04, "Generalization" |
| Nonnegativity of the Milne solution |  | 05, end of the reciprocity section |
| Finite-volume and path-sampling tests |  | 07 |

Rebuild and re-execute everything from the package root with

```bash
python scripts/build_tutorial.py          # all chapters, one to two minutes
python scripts/build_tutorial.py 05 06    # selected chapters
```

The production figure notebooks are in `../notebooks/`.
