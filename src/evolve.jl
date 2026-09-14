# Metropolis Monte Carlo over a whole multi-protein system.
#
# SpinModel's live evolvers (evolvePop, evolvePopStatic) are population
# Wright-Fisher processes over a single sequence. The single-sequence Metropolis
# code (evolve/evolve2) is commented out in SpinModel/src/old/ and references a
# Settings field (ζ) that no longer exists, so the chain-based process this
# project needs is implemented here instead, with ζ as an explicit argument.
#
# RNG: every entry point takes an explicit `rng`. Chains are therefore
# reproducible from a seed without seeding any global stream, which is what makes
# `evolve_ensemble` safe to thread.

"""
    evolve_system([rng], sys, model, assay, n_steps, ζ) -> (final, trajectory)

Metropolis Monte Carlo on a [`ProteinSystem`](@ref) under selective pressure
`assay`. Each step picks one site uniformly across the whole system (all
proteins, all positions), mutates it to a different state, and accepts with
probability `min(1, exp(ζ·Δϕ))`. `ζ` is the inverse evolutionary temperature:
`ζ = 0` is a neutral random walk, large `ζ` is greedy hill climbing.

`sys` is copied, so the caller's system is left untouched. Returns the final
system and the fitness trajectory, a vector of length `n_steps + 1` whose first
entry is the starting fitness.

Only the mutated protein's binding energy is recomputed each step: proteins are
uncoupled, so the other modules' scores are unchanged. This is the pipeline's
hot loop.
"""
function evolve_system(rng::AbstractRNG, sys::ProteinSystem, model::SystemModel,
                       assay::SystemAssay, n_steps::Integer, ζ::Real)
    ζ >= 0 || error("ζ must be non-negative (got $ζ); it is an inverse temperature")
    n_proteins(sys) == n_proteins(model) ||
        error("system has $(n_proteins(sys)) proteins but model has $(n_proteins(model))")

    sys = copy(sys)
    len = sites_per_protein(sys)
    q = model.Q.q
    N = n_proteins(sys)

    ϕ = collect(Float64, module_fitnesses(sys, model))   # per-module cache
    f = system_fitness(assay, ϕ)
    traj = Vector{Float64}(undef, n_steps + 1)
    traj[1] = f

    for t in 1:n_steps
        # Proteins are equal length, so uniform-over-proteins then
        # uniform-within-protein is uniform over all sites.
        p = rand(rng, 1:N)
        i = rand(rng, 1:len)
        seq = sys.seqs[p]

        old_state = seq[i]
        seq[i] = ((old_state - 1 + rand(rng, 1:(q-1))) % q) + 1   # always a different state
        old_ϕp = ϕ[p]
        ϕ[p] = computeFitness(seq, model.Ks[p], model.Q, model.ligs[p], model.assay)
        f_new = system_fitness(assay, ϕ)

        Δ = f_new - f
        if Δ >= 0 || rand(rng) < exp(ζ * Δ)
            f = f_new
        else
            seq[i] = old_state
            ϕ[p] = old_ϕp
        end
        traj[t+1] = f
    end
    return sys, traj
end

evolve_system(sys::ProteinSystem, model::SystemModel, assay::SystemAssay,
              n_steps::Integer, ζ::Real) =
    evolve_system(Random.default_rng(), sys, model, assay, n_steps, ζ)


"""
    evolve_ensemble(founder, model, assay, n_steps, ζ, seeds) -> (systems, trajectories)
    evolve_ensemble(model, assay, n_steps, ζ, seeds)          -> (systems, trajectories)

Run one independent chain per entry of `seeds`, in parallel over threads. The
first form starts every chain from the same `founder` (used to drift a pool of
orthologs from one ancestor); the second draws an independent random start per
chain (used to find a fit founder in the first place).

Each chain gets its own `Xoshiro(seeds[k])`, so results are independent of thread
scheduling and a chain can be replayed on its own from its seed.

Returns `systems::Vector{ProteinSystem}` and `trajectories::Matrix{Float64}` of
size `(n_steps + 1, length(seeds))`, one column per chain.
"""
function evolve_ensemble(founder::ProteinSystem, model::SystemModel, assay::SystemAssay,
                         n_steps::Integer, ζ::Real, seeds::AbstractVector{<:Integer})
    return _evolve_ensemble(rng -> founder, model, assay, n_steps, ζ, seeds)
end

function evolve_ensemble(model::SystemModel, assay::SystemAssay,
                         n_steps::Integer, ζ::Real, seeds::AbstractVector{<:Integer})
    return _evolve_ensemble(rng -> rand_system(rng, model), model, assay, n_steps, ζ, seeds)
end

function _evolve_ensemble(make_start, model::SystemModel, assay::SystemAssay,
                          n_steps::Integer, ζ::Real, seeds::AbstractVector{<:Integer})
    n_chains = length(seeds)
    systems = Vector{ProteinSystem{n_proteins(model)}}(undef, n_chains)
    trajectories = Matrix{Float64}(undef, n_steps + 1, n_chains)

    Threads.@threads for k in 1:n_chains
        rng = Xoshiro(seeds[k])
        start = make_start(rng)
        sys, traj = evolve_system(rng, start, model, assay, n_steps, ζ)
        systems[k] = sys
        @views trajectories[:, k] .= traj
    end
    return systems, trajectories
end
