# Generating and characterizing a chimera library, and the two scans that describe
# the systems it is built from.

"""
    dms_scan(sys, model, assay) -> Matrix{Float64}

Deep mutational scan of one system: a `q x n_sites` matrix whose `(a, i)` entry is
the change in *system* fitness from substituting amino acid `a` at concatenated site
`i`. Wild-type residues score exactly 0.

The per-protein work is `SpinModel.computeBindingDMS`, which scans a protein's single
mutants through a `TransferCache` — rebuilding only the one or two transfer matrices
each mutation touches instead of all `L+1`. It returns the change in the *binding
energy gap*, and `Binding()` scores `E_solvent - E_bound`, so the change in a
protein's fitness is minus that.

Proteins are uncoupled, so a mutation moves exactly one entry of the per-protein
score vector; the system assay is then applied to the updated vector. That last step
cannot be skipped — under `AndGate()` fitness is the minimum over proteins, so a
mutation in the currently-fitter protein may not move system fitness at all.
"""
function dms_scan(sys::ProteinSystem, model::SystemModel, assay::SystemAssay)
    phi = collect(module_fitnesses(sys, model))
    f0 = system_fitness(assay, phi)
    D = Matrix{Float64}(undef, model.Q.q, n_sites(model))
    offset = 0
    for i in 1:n_proteins(model)
        # (q, sites, n_ligands - 1); Binding() uses two conditions, so one slice.
        dE = computeBindingDMS(sys.seqs[i], model.Ks[i], model.Q, model.ligs[i])
        psi = copy(phi)
        for j in axes(dE, 2), a in 1:model.Q.q
            psi[i] = phi[i] - dE[a, j, 1]
            D[a, offset + j] = system_fitness(assay, psi) - f0
        end
        psi[i] = phi[i]
        offset += size(dE, 2)
    end
    return D
end

"""
    chimera_library(model, assay, p1, p2, process; n, seed, n_chunks) -> NamedTuple

Shuffle two parent sequences into `n` chimeras, score each, and record the
ground-truth ancestry from the generating process. Returns `sequences` (`n_sites x
n`, `Int8`), `ancestry` (`BitMatrix`, true = parent 1), `fitness` and `crossovers`.

Reproducibility does not depend on the thread count: work is split into a fixed
`n_chunks`, each with its own seeded RNG, so the same `seed` gives the same library
on any machine and at any `Threads.nthreads()`.
"""
function chimera_library(model::SystemModel, assay::SystemAssay,
                         p1::AbstractVector, p2::AbstractVector,
                         process::CrossoverProcess;
                         n::Integer, seed::Integer, n_chunks::Integer=100)
    total_sites = n_sites(model)
    np = n_proteins(model)
    sequences = Matrix{Int8}(undef, total_sites, n)
    # Bool, not BitMatrix: 110 rows means adjacent columns share a 64-bit word, so
    # threads writing neighbouring columns of a BitMatrix would race. Packed below.
    ancestry_bytes = Matrix{Bool}(undef, total_sites, n)
    fitness = Vector{Float64}(undef, n)
    crossovers = Vector{Int16}(undef, n)

    bounds = round.(Int, range(0, n, length=n_chunks + 1))
    Threads.@threads for c in 1:n_chunks
        rng = Xoshiro(seed + c)
        anc = Vector{Bool}(undef, total_sites)
        flat = Sequence(undef, total_sites)
        for i in (bounds[c] + 1):bounds[c + 1]
            draw_ancestry!(anc, rng, process)
            apply_ancestry!(flat, anc, p1, p2)
            fitness[i] = system_fitness(assay, split_system(flat, np), model)
            @views sequences[:, i] .= flat
            @views ancestry_bytes[:, i] .= anc
            crossovers[i] = n_crossovers(anc)
        end
    end

    ancestry = BitMatrix(ancestry_bytes)
    ancestry_bytes = nothing        # ~110 MB at n = 1e6; let it go before the caller saves
    GC.gc()
    return (; sequences, ancestry, fitness, crossovers)
end

"""
    site_conservation(sequences, q) -> (entropies, conservation)

Per-site Shannon entropy over the `q` states, in bits, and conservation
`log2(q) - H`: a site fixed across every sequence scores `log2(q)`, one uniform over
the alphabet scores 0. Plug-in estimator — with `n` sequences its downward bias on
`H` is about `(q-1)/(2n ln2)` bits, so it is worth checking against the pool size
rather than correcting for.
"""
function site_conservation(sequences::AbstractMatrix, q::Integer)
    n_sites_, n = size(sequences)
    entropies = Vector{Float64}(undef, n_sites_)
    counts = zeros(Int, q)
    for i in 1:n_sites_
        fill!(counts, 0)
        for j in 1:n
            counts[sequences[i, j]] += 1
        end
        entropies[i] = -sum(c > 0 ? (c / n) * log2(c / n) : 0.0 for c in counts)
    end
    return entropies, log2(q) .- entropies
end

"""
    pairwise_divergence(sequences) -> NamedTuple

Hamming distances between every pair of columns, the same thing as sequence
similarity (`1 - d / n_sites`), plus the upper-triangle distances and their mean as
a fraction of the sequence length. Two parents that barely differ leave the ancestry
analysis nothing to see, so this is the number that decides whether a pool is usable.
"""
function pairwise_divergence(sequences::AbstractMatrix)
    total_sites, n = size(sequences)
    distances = [sum(sequences[:, i] .!= sequences[:, j]) for i in 1:n, j in 1:n]
    similarities = 1 .- distances ./ total_sites
    upper = [distances[i, j] for i in 1:n for j in (i + 1):n]
    mean_divergence = mean(upper) / total_sites
    return (; distances, similarities, upper, mean_divergence,
              mean_similarity = 1 - mean_divergence)
end

"""
    choose_parent_pair(n_pool, pair, seed) -> Tuple{Int,Int}

The two orthologs the library is built from: `pair` if given, otherwise a random
distinct pair from `seed`. Divergence across a drifted pool is uniform, so a random
pair is representative — but the choice is a parameter, not a constant, so the
pipeline can be re-run across pairs to test whether inferred modularity is robust.
"""
function choose_parent_pair(n_pool::Integer, pair, seed::Integer)
    pair === nothing || return Tuple(pair)
    rng = Xoshiro(seed)
    a = rand(rng, 1:n_pool)
    b = rand(rng, setdiff(1:n_pool, a))
    return (a, b)
end

"""
    crossover_bins(crossovers; min_count, max_bins) -> Vector{Int}

The crossover counts worth plotting or tabulating: those carrying at least `min_count`
chimeras, thinned to at most `max_bins` evenly spaced values.

Derived from the data rather than fixed at 0:12, because the range moves with the
crossover rate — at p = 0.03 a chimera has ~3 crossovers and at p = 0.8 it has ~87, so a
fixed window is either most of the distribution or none of it. An empty window is not
merely uninformative: a per-count mean or percentile over it is undefined.
"""
function crossover_bins(crossovers::AbstractVector; min_count::Integer=20,
                        max_bins::Integer=30)
    present = [k for k in minimum(crossovers):maximum(crossovers)
               if count(==(k), crossovers) >= min_count]
    isempty(present) && return [Int(round(median(crossovers)))]
    length(present) <= max_bins && return present
    idx = round.(Int, range(1, length(present), length=max_bins))
    return present[unique(idx)]
end
