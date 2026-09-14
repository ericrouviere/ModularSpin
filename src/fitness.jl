# System-level fitness: maps the vector of per-protein binding energies to one
# scalar. Follows the pattern of SpinModel/src/fitness.jl -- one immutable
# singleton per selective pressure, dispatched as a value, so adding a pressure
# is a new struct plus one method rather than an edit to a central branch.

"""
    SystemAssay

Selective pressure acting on a whole [`ProteinSystem`](@ref). Subtypes are
singletons passed as values, e.g. `system_fitness(AndGate(), sys, model)`.
"""
abstract type SystemAssay end

"""
    AndGate()

Every module must work: fitness is the *minimum* per-protein binding score. A
broken module cannot be rescued by a strong one, so functional chimeras must be
self-consistent within each module.
"""
struct AndGate <: SystemAssay end

"""
    Compensatory()

Modules trade off: fitness is the *sum* of per-protein binding scores, so a
strong module can compensate for a weak one. The contrast with [`AndGate`](@ref)
is what makes module structure detectable (or not) downstream.
"""
struct Compensatory <: SystemAssay end

system_fitness(::AndGate, ϕ) = minimum(ϕ)
system_fitness(::Compensatory, ϕ) = sum(ϕ)

"""
    module_fitnesses(sys, model) -> NTuple{N,Float64}

Per-protein score under the model's per-protein assay (`model.assay`), computed
independently for each protein via `SpinModel.computeFitness`. Each protein uses
its own table `K` and its own ligands. Positive = better; for the binding assays
the score is a free-energy difference against the solvent reference, so tighter
binding is larger.
"""
module_fitnesses(sys::ProteinSystem{N}, model::SystemModel{N}) where {N} =
    ntuple(i -> computeFitness(sys.seqs[i], model.Ks[i], model.Q, model.ligs[i], model.assay), N)

"""
    system_fitness(assay, sys, model) -> Float64

Scalar fitness of a whole system under `assay`.
"""
system_fitness(assay::SystemAssay, sys::ProteinSystem, model::SystemModel) =
    system_fitness(assay, module_fitnesses(sys, model))

"""
    system_fitness(assay, systems, model) -> Vector{Float64}

Fitness of many systems, one thread-parallel pass. Used for the ~10^6 chimera
library, where the free-energy evaluations dominate the whole pipeline's runtime.
"""
function system_fitness(assay::SystemAssay, systems::AbstractVector{<:ProteinSystem},
                        model::SystemModel)
    fits = Vector{Float64}(undef, length(systems))
    Threads.@threads for i in eachindex(systems)
        fits[i] = system_fitness(assay, systems[i], model)
    end
    return fits
end
