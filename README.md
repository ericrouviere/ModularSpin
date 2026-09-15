# ModularSpin

Multi-protein spin-glass systems with a ground-truth modular decomposition, and the
machinery to shuffle them into chimeras and ask whether the modules can be inferred back out.

A **system** is a vector of *uncoupled* spin-glass proteins built on
[`SpinModel`](https://github.com/ericrouviere/SpinModel). Because there are no couplings
between proteins, the protein boundaries are the ground-truth module boundaries by
construction. The pipeline:

1. `SystemModel` — one independent coupling table per protein, shared `Settings`, per-protein `Ligands`
2. `SystemAssay` (`AndGate`, `Compensatory`) — maps per-protein binding energies to one system fitness
3. `evolve_system` / `evolve_ensemble` — Metropolis Monte Carlo that evolves a fit founder and drifts it into diverged orthologs
4. `CrossoverProcess` (`BernoulliCrossover`) — recombines two orthologs into chimeras with recorded ancestry
5. Downstream analysis — chimera libraries, ancestry moments, regression/neural-net models, and clustering of inferred interactions

## Requirements

- Julia ≥ 1.10
- [`SpinModel`](https://github.com/ericrouviere/SpinModel), which is not in the General
  registry and must be added by URL (below)
- `PyPlot` uses Python's matplotlib through PyCall; by default PyCall installs its own
  Conda-based Python on first build, so no manual setup is usually needed

SpinModel is currently a private repository, so installing it requires a GitHub account with
access. Julia's built-in git library cannot use your stored GitHub credentials and stops at a
username prompt, so tell Pkg to use the system `git` instead, which picks up credentials from
`gh auth login` or the macOS keychain:

```bash
export JULIA_PKG_USE_CLI_GIT=true   # add to ~/.zshrc to make it permanent
```

## Installation

Neither package is registered, so add them by URL. **SpinModel must be added first** —
otherwise Pkg cannot resolve ModularSpin's dependency on it.

### Use it in a project

With `JULIA_PKG_USE_CLI_GIT=true` set in the shell that launches Julia (see Requirements),
start the Julia REPL in the environment you want to use (ideally a project environment for
your analysis rather than the global one):

```julia
using Pkg
Pkg.activate(".")   # your analysis project
Pkg.add(url="https://github.com/ericrouviere/SpinModel")
Pkg.add(url="https://github.com/ericrouviere/ModularSpin")
```

Equivalently, in Pkg mode (press `]`):

```
pkg> activate .
pkg> add https://github.com/ericrouviere/SpinModel
pkg> add https://github.com/ericrouviere/ModularSpin
```

To pull later changes from GitHub, run `Pkg.update()`.

### Develop the package

To edit ModularSpin itself, clone it and let Pkg track the local checkout instead of a
pinned commit:

```bash
git clone https://github.com/ericrouviere/ModularSpin.git
cd ModularSpin
JULIA_PKG_USE_CLI_GIT=true julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/ericrouviere/SpinModel"); Pkg.instantiate()'
```

If you also have a local clone of SpinModel, use `Pkg.develop(path="path/to/SpinModel")` in
place of `Pkg.add(url=...)`. Changes to ModularSpin can then be used from any other project
with `Pkg.develop(path="path/to/ModularSpin")`.

Run the test suite (a few seconds):

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Loading it in scripts

Put the package in a project environment (see above) and activate that environment at the top
of the script, or launch Julia with `julia --project=path/to/project script.jl`:

```julia
using Pkg
Pkg.activate(@__DIR__)   # the project containing ModularSpin; omit if using --project

using ModularSpin        # system types, assays, evolution, crossover, analysis
using SpinModel          # Settings, Ligands, Perturbation, ... used to configure a model
using Random
using LinearAlgebra

# A two-protein system: W×L lattice, q states, one binding site on the last column
W, L, q = 4, 5, 5
Q = Settings(W, L, q)
rng = Xoshiro(1)
site = [CartesianIndex(i, L + 1, 1) for i in 1:W]
ligs = Ligands([Perturbation(site, zeros(W)),                 # solvent
                Perturbation(site, normalize(randn(rng, W)))]) # ligand
model = SystemModel(Q, ligs, 2; assay=Binding(), rng)
```

Every stochastic entry point takes an explicit `rng::AbstractRNG`; nothing seeds the global
stream, so pass a seeded RNG (e.g. `Xoshiro(seed)`) to make a run reproducible.

During interactive development, load `Revise` before the package so edits to the source are
picked up without restarting Julia:

```julia
using Revise
using ModularSpin
```

Function names in `snake_case` come from ModularSpin; `camelCase` names come from SpinModel.
See `?ModularSpin` and the docstrings of individual functions (e.g. `?evolve_system`) for
details.
