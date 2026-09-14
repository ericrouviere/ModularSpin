# A fixed random relabelling of sites, so that a module is a SET of positions rather
# than a contiguous stretch of them.
#
# Why this exists. In this model the two proteins are concatenated, so module membership
# and sequence position are the same fact and every analysis downstream can lean on
# adjacency without anyone noticing. A real system does not oblige: the residues that
# form a functional unit are scattered through the chain. Relabelling the sites with a
# fixed permutation, applied AFTER the library has been generated and scored, makes the
# inference problem the realistic one while changing nothing about the physics — the
# fitness was measured on the true system, and only the representation moves.
#
# What it should and should not change. Almost everything downstream is
# permutation-equivariant: a covariance matrix, a linear model on independent site
# features, a network with no notion of order, and a clustering of sites all give the
# same answer up to relabelling. So the numbers should be unmoved by this. What it does
# break is anything that reads structure off ADJACENCY -- the AR(1) check on the
# crossover generator, any window statistic, and the eye's habit of reading blocks off a
# heatmap. Those are exactly the places where an analysis was getting the answer from
# the layout rather than from the data, which is what makes the scramble worth running.

"""
    scramble_permutation(n, seed) -> Vector{Int}

A fixed random relabelling of `1:n`. Entry `k` names the TRUE site that scrambled
position `k` holds, so scrambled data is `data[perm, :]` and `perm` is what any result
has to be pushed back through to be read on the structure.
"""
scramble_permutation(n::Integer, seed::Integer) = randperm(Xoshiro(seed), n)

"""
    scramble_rows(M, perm) -> Matrix

Relabel the ROWS (sites) of a site-by-sample matrix. Columns — chimeras — are untouched.
"""
scramble_rows(M::AbstractMatrix, perm::AbstractVector{<:Integer}) = M[perm, :]

"""
    unscramble(v, perm) -> Vector

Push a per-site vector from scrambled positions back to true site indices, the inverse of
`scramble_rows`: entry `perm[k]` of the result is entry `k` of the input. This is what a
mode, a field, or a cluster label has to pass through before it can be drawn on the
structure.
"""
function unscramble(v::AbstractVector, perm::AbstractVector{<:Integer})
    out = similar(v)
    out[perm] = v
    return out
end

"""
    unscramble_rows(M, perm) -> Matrix

The same for a site-by-sample matrix: rows go back to true site indices, columns are
untouched.
"""
function unscramble_rows(M::AbstractMatrix, perm::AbstractVector{<:Integer})
    out = similar(M)
    out[perm, :] = M
    return out
end

"""
    unscramble_both(M, perm) -> Matrix

The same for a site-by-SITE matrix, where both axes are sites: a covariance, a precision,
an interaction matrix. Named apart from `unscramble_rows` deliberately — the two differ
only in the shape of their argument, and a site-by-sample matrix silently permuted on both
axes would be a wrong answer rather than an error.
"""
function unscramble_both(M::AbstractMatrix, perm::AbstractVector{<:Integer})
    size(M, 1) == size(M, 2) == length(perm) ||
        error("unscramble_both expects a site-by-site matrix, got $(size(M)) for $(length(perm)) sites")
    out = similar(M)
    out[perm, perm] = M
    return out
end

"""
    scrambled_module_labels(perm, boundary) -> Vector{Int}

Which true module each SCRAMBLED position belongs to: 1 for a site from the first
protein, 2 otherwise. This is the ground truth in the coordinates the analysis works in,
and it replaces the boundary index everywhere a statistic needs to know what is a module.
"""
scrambled_module_labels(perm::AbstractVector{<:Integer}, boundary::Integer) =
    [p <= boundary ? 1 : 2 for p in perm]
