# Core data structures for multi-protein spin-glass systems.
#
# A "system" is a vector of independent spin-glass proteins. There are no
# couplings between proteins, so the modular decomposition is imposed by
# construction: protein boundaries *are* the ground-truth module boundaries.

"""
    ProteinSystem{N}

A system of `N` proteins, each an independent spin sequence (`SpinModel.Sequence`,
a `Vector{Int8}` of length `W*(L+1)`).

The struct is immutable, so a system cannot be repointed at different sequences.
The underlying vectors are still mutable; only the evolver mutates them, and only
on a copy it owns (see `evolve_system`).

Construct from a tuple of `Sequence`s, or from any vector of integer vectors:

    ProteinSystem((seq1, seq2))
    ProteinSystem([seq1, seq2])
"""
struct ProteinSystem{N}
    seqs::NTuple{N, Sequence}
end

ProteinSystem(seqs::AbstractVector{<:AbstractVector}) =
    ProteinSystem(ntuple(i -> Sequence(seqs[i]), length(seqs)))

Base.:(==)(a::ProteinSystem, b::ProteinSystem) = a.seqs == b.seqs
Base.copy(sys::ProteinSystem) = ProteinSystem(map(copy, sys.seqs))
Base.getindex(sys::ProteinSystem, i::Integer) = sys.seqs[i]

"""
    n_proteins(x)

Number of proteins (modules) in a system or model.
"""
n_proteins(::ProteinSystem{N}) where {N} = N

"""
    n_sites(x)

Total number of spins across all proteins.
"""
n_sites(sys::ProteinSystem) = sum(length, sys.seqs)

"""
    sites_per_protein(x)

Number of spins in each protein. Errors unless all proteins are the same length,
which the rest of the pipeline assumes (all proteins share one `Settings`).
"""
function sites_per_protein(sys::ProteinSystem)
    len = length(sys.seqs[1])
    all(s -> length(s) == len, sys.seqs) ||
        error("proteins have unequal lengths $(map(length, sys.seqs)); the pipeline assumes a shared Settings")
    return len
end

"""
    concatenate(sys) -> Sequence

Flatten the system into one sequence, proteins laid end to end (protein 1 first).
This is the representation crossovers and Hamming distances operate on, so a
crossover point may fall inside a protein or exactly on a protein boundary.
"""
concatenate(sys::ProteinSystem) = reduce(vcat, sys.seqs)

"""
    split_system(flat, n) -> ProteinSystem

Inverse of [`concatenate`](@ref): cut a flat sequence into `n` equal contiguous
blocks. Copies, so the result does not alias `flat`.
"""
function split_system(flat::AbstractVector, n::Integer)
    len, rem = divrem(length(flat), n)
    rem == 0 || error("flat sequence of length $(length(flat)) does not divide into $n equal proteins")
    return ProteinSystem(ntuple(i -> Sequence(flat[(i-1)*len+1 : i*len]), n))
end

"""
    hamming(a, b)

Number of positions at which two systems (or two flat sequences) differ.
"""
hamming(a::AbstractVector, b::AbstractVector) = sum(x != y for (x, y) in zip(a, b))
hamming(a::ProteinSystem, b::ProteinSystem) = sum(hamming(x, y) for (x, y) in zip(a.seqs, b.seqs))


"""
    SystemModel{N,A}

The physical model behind a [`ProteinSystem`](@ref): one random table `K` per
protein, one shared `Settings`, one `Ligands` per protein, and the
`SpinModel.Assay` scoring each protein on its own.

`K` is the sequence-to-coupling map — the amino-acid chemistry — so whether the
proteins share one is a modelling choice, made at construction with
`shared_table`. Sharing says the chemistry is universal and the proteins differ
only in sequence and in what they bind; drawing independently says each protein
also has its own chemistry.

Orthologs share a `SystemModel` and differ only in sequence.

The per-protein `assay` and `ligs` must agree on ligand count: `Binding()` needs
2 perturbations (solvent + one ligand), `DoubleBinding()` and `Specificity()`
need 3, `Allostery()` needs 4.

The choice also sets how spread out the functional unit is, which is worth
knowing when interpreting any downstream inference. `Binding()` is unfrustrated:
a protein saturates it by polarizing its binding site, so the functional unit
concentrates there (measured on `W=5, L=10`: 44-70% of single-mutant fitness
effect on the 10 binding-site spins, ~20 of 110 sites with any effect at all).
`DoubleBinding()`, a generalist binding two different ligands at once, cannot be
satisfied locally and spreads the unit across the lattice (12-18% at the binding
site, 60-90 sites contributing).
"""
struct SystemModel{N, A<:Assay}
    Ks::NTuple{N, Table}
    Q::Settings
    ligs::NTuple{N, Ligands}
    assay::A
end

n_proteins(::SystemModel{N}) where {N} = N
sites_per_protein(model::SystemModel) = model.Q.W * (model.Q.L + 1)
n_sites(model::SystemModel) = n_proteins(model) * sites_per_protein(model)

"""
    SystemModel(Q, ligs, n; assay, sh, sJ, rng)

Build an `n`-protein model. `ligs` is either one `Ligands` shared by every protein
or a collection of `n` of them. `assay` is the per-protein selective pressure
(default `Binding()`; see the note on `SystemModel`). `sh` and `sJ` are the field
and coupling standard deviations, as in `SpinModel.rand_table`.

`shared_table=true` draws one table and hands the same one to every protein — the
amino-acid chemistry is universal, and the proteins differ only in sequence and
ligands. The default draws an independent table per protein. When shared, the
tuple entries alias a single array, which is safe because nothing here mutates a
table; copy it first if you intend to.
"""
SystemModel(Q::Settings, ligs::Ligands, n::Integer; kwargs...) =
    SystemModel(Q, ntuple(_ -> ligs, n), n; kwargs...)

function SystemModel(Q::Settings, ligs, n::Integer;
                     assay::Assay=Binding(), shared_table::Bool=false,
                     sh::Real=3.0, sJ::Real=3.0, rng::AbstractRNG=Random.default_rng())
    length(ligs) == n || error("got $(length(ligs)) Ligands for $n proteins")
    for (i, lig) in enumerate(ligs)
        _check_ligand_count(assay, lig, i)
    end
    Ks = if shared_table
        K = _rand_table(rng, Q, sh, sJ)
        ntuple(_ -> K, n)
    else
        ntuple(_ -> _rand_table(rng, Q, sh, sJ), n)
    end
    return SystemModel(Ks, Q, Tuple(ligs), assay)
end

# SpinModel's scoring functions splat the free-energy vector, so a mismatch shows
# up as a confusing MethodError deep in computeFitness. Catch it at construction.
const _N_LIGANDS = Dict(Stability => 1, Binding => 2, Binding1 => 3, Binding2 => 3,
                        DoubleBinding => 3, Specificity => 3,
                        Allostery => 4, NegativeAllostery => 4)

function _check_ligand_count(assay::Assay, lig::Ligands, i::Integer)
    want = get(_N_LIGANDS, typeof(assay), nothing)
    want === nothing && return nothing
    length(lig) == want ||
        error("$(typeof(assay)) needs $want ligand conditions but protein $i has $(length(lig))")
    return nothing
end

# Mirrors SpinModel.rand_table (same construction, same K layout) but takes an
# explicit RNG, so a model is reproducible from a stream without seeding globals.
function _rand_table(rng::AbstractRNG, Q::Settings, sh::Real, sJ::Real)
    K = randn(rng, 3, Q.W, Q.L + 1, Q.q^2)
    K[1, :, :, :] .*= sh
    K[2:3, :, :, :] .*= sJ
    return K
end

"""
    rand_system([rng], model) -> ProteinSystem

A system of independent uniformly random sequences, one per protein.
"""
rand_system(model::SystemModel) = rand_system(Random.default_rng(), model)

function rand_system(rng::AbstractRNG, model::SystemModel)
    len = sites_per_protein(model)
    q = model.Q.q
    return ProteinSystem(ntuple(_ -> Sequence(rand(rng, 1:q, len)), n_proteins(model)))
end
