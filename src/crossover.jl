# DNA-shuffling-like recombination between two parent systems.
#
# Crossovers act on the *concatenated* system (proteins laid end to end), so a
# breakpoint may fall inside a protein or exactly on a protein boundary --
# nothing in the generator knows where the modules are.
#
# Ancestry is recorded directly from the generating process, never reconstructed
# from the chimeric sequence: the two parents share a residue at most sites, so
# post-hoc recovery is impossible there.

"""
    CrossoverProcess

A stochastic process generating parent-of-origin labels along a concatenated
sequence. Implement a new process by subtyping this and defining one method,
[`draw_ancestry!`](@ref); everything downstream (chimera construction, the
library script, the analyses) goes through that interface unchanged.
"""
abstract type CrossoverProcess end

"""
    BernoulliCrossover(p)

The sequence is discrete, so the process is defined directly on it: each of the
`n - 1` gaps between adjacent sites independently carries a crossover with
probability `p`. Expected crossover count over `n` sites is exactly `p * (n - 1)`.

Ancestry is then a two-state Markov chain along the sequence — a fair coin at the
first site, flipped at every crossover. Spacing between crossovers is geometric
with mean `1/p`, the count in any window is binomial, and

    ⟨sᵢsⱼ⟩ = (1 - 2p)^|i-j|,    sᵢ = ±1

a correlation structure present even with no selection at all, which is why the
analysis needs an unselected null.

`p` is a probability, so it is capped at 1: `p = 0.5` makes sites i.i.d. and
`p = 1` alternates strictly.
"""
struct BernoulliCrossover <: CrossoverProcess
    p::Float64

    function BernoulliCrossover(p::Real)
        0 <= p <= 1 || error("crossover probability must lie in [0, 1], got $p")
        return new(Float64(p))
    end
end

"""
    draw_ancestry!(ancestry, rng, process) -> ancestry

Fill `ancestry` with parent-of-origin labels (`true` = parent 1, `false` =
parent 2). Which parent contributes the first segment is a fair coin flip.

This is the one method a `CrossoverProcess` must implement, and it is the hot
path of the ~10^6-chimera library, so it allocates nothing.
"""
function draw_ancestry!(ancestry::AbstractVector{Bool}, rng::AbstractRNG,
                        process::BernoulliCrossover)
    n = length(ancestry)
    n == 0 && return ancestry

    current = rand(rng, Bool)
    ancestry[1] = current
    p = process.p
    p > 0 || (fill!(ancestry, current); return ancestry)

    # One Bernoulli(p) trial per gap: a crossover flips the parent for the rest of
    # the sequence. Nothing lives between sites, so a crossover either shows up in
    # the ancestry or does not happen.
    for i in 2:n
        rand(rng) < p && (current = !current)
        ancestry[i] = current
    end
    return ancestry
end

draw_ancestry!(ancestry::AbstractVector{Bool}, process::CrossoverProcess) =
    draw_ancestry!(ancestry, Random.default_rng(), process)

"""
    crossover_points(ancestry) -> Vector{Int}

Positions at which the parent of origin switches: `i` is listed when
`ancestry[i] != ancestry[i-1]`. With one trial per gap these are exactly the
crossovers drawn, so the count is not an undercount of anything latent.
"""
crossover_points(ancestry::AbstractVector{Bool}) =
    [i for i in 2:length(ancestry) if ancestry[i] != ancestry[i-1]]

"""
    n_crossovers(ancestry) -> Int

Number of observed parent switches, i.e. `length(crossover_points(ancestry))`
without building the vector.
"""
n_crossovers(ancestry::AbstractVector{Bool}) =
    count(i -> ancestry[i] != ancestry[i-1], 2:length(ancestry))

"""
    apply_ancestry!(flat, ancestry, parent1, parent2) -> flat

Write the chimeric sequence implied by `ancestry` into `flat`, taking site `i`
from `parent1` where `ancestry[i]` is true and from `parent2` otherwise. All four
arguments are flat (concatenated) sequences of the same length.
"""
function apply_ancestry!(flat::AbstractVector, ancestry::AbstractVector{Bool},
                         parent1::AbstractVector, parent2::AbstractVector)
    flat .= ifelse.(ancestry, parent1, parent2)
    return flat
end

"""
    chimera([rng], process, parent1, parent2, n_proteins) -> (sys, ancestry)

One chimera from two flat parent sequences: draw ancestry from `process`, then
realize the sequence. Returns the chimeric [`ProteinSystem`](@ref) and its
ground-truth ancestry.

The flat-parent method is the one to call in a loop — the `ProteinSystem` method
below concatenates its parents on every call.
"""
function chimera(rng::AbstractRNG, process::CrossoverProcess,
                 parent1::AbstractVector, parent2::AbstractVector, n_proteins::Integer)
    length(parent1) == length(parent2) ||
        error("parents have unequal lengths $(length(parent1)) and $(length(parent2))")
    ancestry = BitVector(undef, length(parent1))
    draw_ancestry!(ancestry, rng, process)
    flat = Sequence(undef, length(parent1))
    apply_ancestry!(flat, ancestry, parent1, parent2)
    return split_system(flat, n_proteins), ancestry
end

chimera(rng::AbstractRNG, process::CrossoverProcess, a::ProteinSystem, b::ProteinSystem) =
    chimera(rng, process, concatenate(a), concatenate(b), n_proteins(a))

chimera(process::CrossoverProcess, args...) =
    chimera(Random.default_rng(), process, args...)

"""
    polymorphic_sites(parent1, parent2) -> Vector{Int}

Indices where the two flat parent sequences differ. Sites where the parents agree
carry an ancestry label but have no effect on fitness, so they contribute pure
crossover-process correlation with no functional signal and are excluded from the
inference in Step 5.
"""
polymorphic_sites(parent1::AbstractVector, parent2::AbstractVector) =
    findall(parent1 .!= parent2)

polymorphic_sites(a::ProteinSystem, b::ProteinSystem) =
    polymorphic_sites(concatenate(a), concatenate(b))
