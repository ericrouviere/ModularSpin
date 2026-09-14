# Turning a scored chimera library into a supervised dataset: encoding, the
# deduplication the library's heavy duplication forces, and leak-free splits.

"""
    functional_cutoff(fitness, quantile) -> Float64

The raw fitness at which a chimera is called functional: a quantile of the library's
own fitness distribution, so the top `1 - quantile` of the library is functional.

A quantile of the library rather than a fitness value, because the fitness scale
moves with the ligands, the assay and the parents while the selected fraction does
not. A quantile of the library rather than of the random-system distribution,
because recombining two fit parents never produces anything close to random — the
entire library clears even the fittest random system, so a "better than random" bar
cannot split it. Take it over the library as generated, so every step downstream
splits on one common cutoff.
"""
functional_cutoff(fitness::AbstractVector, quantile::Real) =
    StatsBase.quantile(fitness, quantile)

"""
    spin_encode(sequences) -> (X, reference)

Encode each site as a spin, `+1` or `-1`. The two residues present at a site are read
off the data; the larger is called the reference and encoded as `+1`. Sites carrying
only one residue encode as all `-1` — constant, but retained, so feature index and site
index stay the same thing.

Spins rather than bits, because the model fitted on this design is read as an Ising
model. With `x ∈ {0,1}` a pairwise term `β x_i x_j` fires only in the (1,1) corner and
its three other corners are absorbed into the intercept and the site terms, so a
coefficient answers "what does this one combination add" rather than "do these two
sites prefer to agree". With `s ∈ {-1,+1}` the term `β s_i s_j` is symmetric: positive
favours agreement, negative favours disagreement, and the site terms are fields in the
same sense. The two codings describe the same fitted function — `s = 2x - 1` maps one
to the other with the interaction coefficient's SIGN preserved — but only one of them
splits that function into parameters that mean what their names say.

Deliberately derived from sequence alone. It coincides with ancestry up to an arbitrary
per-site flip, which is what makes "withhold ancestry" an honest framing rather than a
real restriction: with two parents the two are the same variable. That flip is the one
thing spins do NOT fix — `+1` is parent 1's residue at some sites and parent 2's at
others, so a coupling's sign says whether the two sites prefer to agree *in this
coding*, not whether they prefer to come from the same parent. Re-orienting per site
against a parent would answer that, and would use the ancestry this encoding is meant
to withhold.
"""
function spin_encode(sequences::AbstractMatrix)
    n_sites_, n = size(sequences)
    X = Matrix{Int8}(undef, n_sites_, n)
    reference = zeros(eltype(sequences), n_sites_)
    Threads.@threads for i in 1:n_sites_
        states = sort!(unique(@view sequences[i, :]))
        length(states) <= 2 ||
            error("site $i has $(length(states)) states; expected <= 2 for a 2-parent library")
        reference[i] = states[end]
        @views X[i, :] .= ifelse.(sequences[i, :] .== states[end], Int8(1), Int8(-1))
    end
    return X, reference
end

"""
    encoding_matches_ancestry(X, ancestry) -> Int

Number of varying sites whose encoded spin fails to match ancestry up to a per-site
flip. Zero is the only acceptable value; this is a test of the encoder, not an input
to it. Constant sites carry no ancestry information and are exempt.
"""
function encoding_matches_ancestry(X::AbstractMatrix, ancestry::AbstractMatrix)
    bad, checked = 0, 0
    for i in axes(X, 1)
        varying(v) = any(x -> x > 0, v) && any(x -> x < 0, v)
        varying(@view X[i, :]) || continue
        checked += 1
        agree = count((X[i, j] > 0) == ancestry[i, j] for j in axes(X, 2))
        (agree == size(X, 2) || agree == 0) || (bad += 1)
    end
    return bad, checked
end

"""
    group_by_sequence(X, fitness, crossovers) -> NamedTuple

Group library rows by identical encoded sequence. Two different ancestry patterns
give the same sequence whenever their breakpoints differ only at constant sites, so a
group can span several crossover counts; it is labelled with the smallest, i.e. the
most parsimonious number of crossovers that could have produced it.

Returns the representative row of each group, the group of every row, the
multiplicities and the per-group crossover label. Duplication is heavy and unevenly
distributed — concentrated in the lightly recombined rows — which is why every split
downstream is made over groups and never over rows.
"""
function group_by_sequence(X::AbstractMatrix, crossovers::AbstractVector)
    n_rows = size(X, 2)
    groups = Dict{UInt64,Int32}()
    group_of_row = Vector{Int32}(undef, n_rows)
    representative = Int32[]
    for j in 1:n_rows
        h = hash(@view X[:, j])
        g = get(groups, h, Int32(0))
        if g == 0
            push!(representative, j)
            g = Int32(length(representative))
            groups[h] = g
        end
        group_of_row[j] = g
    end
    n_groups = length(representative)
    multiplicity = zeros(Int32, n_groups)
    group_crossovers = fill(typemax(Int16), n_groups)
    for j in 1:n_rows
        g = group_of_row[j]
        multiplicity[g] += 1
        group_crossovers[g] = min(group_crossovers[g], crossovers[j])
    end
    return (; representative, group_of_row, multiplicity, group_crossovers, n_groups)
end

"""
    fitness_spread_within_groups(fitness, group_of_row, representative) -> Float64

Largest disagreement between a row's fitness and its group representative's. Fitness
is a deterministic function of sequence, so this must be zero to numerical precision;
anything else means the encoding has collapsed sequences that differ.
"""
function fitness_spread_within_groups(fitness, group_of_row, representative)
    fit_of_group = fitness[representative]
    worst = 0.0
    for j in eachindex(fitness)
        worst = max(worst, abs(fitness[j] - fit_of_group[group_of_row[j]]))
    end
    return worst
end

"""
    stratified_split(rng, strata, fractions) -> Vector{Int8}

Assign each item a split label (1 = train, 2 = validation, 3 = test) by shuffling
within each stratum and cutting at `fractions`. Stratifying on crossover count keeps
every recombination depth represented in all three splits, which matters because the
depths differ enormously in size.
"""
function stratified_split(rng::AbstractRNG, strata::AbstractVector, fractions)
    label = Vector{Int8}(undef, length(strata))
    for s in unique(strata)
        idx = shuffle(rng, findall(==(s), strata))
        n = length(idx)
        n_train = round(Int, fractions[1] * n)
        n_val = round(Int, fractions[2] * n)
        label[idx[1:n_train]] .= 1
        label[idx[(n_train + 1):(n_train + n_val)]] .= 2
        label[idx[(n_train + n_val + 1):end]] .= 3
    end
    return label
end

"""
    random_split(rng, n, fractions) -> Vector{Int8}

An unstratified split over `n` items. Used for the deliberately broken control: a
split over library ROWS, which puts the same sequence on both sides of the boundary
and so measures what duplicate contamination buys.
"""
function random_split(rng::AbstractRNG, n::Integer, fractions)
    label = Vector{Int8}(undef, n)
    idx = shuffle(rng, 1:n)
    n_train = round(Int, fractions[1] * n)
    n_val = round(Int, fractions[2] * n)
    label[idx[1:n_train]] .= 1
    label[idx[(n_train + 1):(n_train + n_val)]] .= 2
    label[idx[(n_train + n_val + 1):end]] .= 3
    return label
end

"""
    leak_fraction(naive_label, group_of_row) -> Float64

Share of test rows under a row-based split whose exact sequence also appears in
training — the leak that split-over-groups exists to prevent, measured rather than
asserted.
"""
function leak_fraction(naive_label::AbstractVector, group_of_row::AbstractVector)
    train_groups = Set(group_of_row[naive_label .== 1])
    test_rows = findall(==(3), naive_label)
    return count(j -> group_of_row[j] in train_groups, test_rows) / length(test_rows)
end
