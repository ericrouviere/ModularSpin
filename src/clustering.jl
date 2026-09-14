# Hierarchical clustering of a fitted interaction matrix, and the statistics that say
# whether a tree found anything.
#
# A module is defined by the ABSENCE of cross-boundary coupling, while clustering
# finds groups by the PRESENCE of within-group coupling: it will faithfully return the
# set of sites that interact, whether or not that set is a module. Nothing here can
# close that gap, so the callers report every cluster with the share of total |J| it
# contains, and a cluster of sites that interact with nothing is labelled as such.
#
# Two things a tree can get right, and they are not the same. A cut is a partition; a
# tree also fixes an ORDER, and the order can carry structure that no cut of the same
# tree expresses -- a top-level merge that peels off one isolated site leaves every
# real grouping below it. `adjusted_rand` scores the cut, `run_test` scores the order.

"""
    coupling_distance(J) -> Matrix

The dissimilarity of the header: `1 - |J_ij| / max|J|`, with a zero diagonal. Scaled
by the largest coupling so the tree's heights are in [0, 1] and the ridge and lasso
trees can be drawn on one axis -- a monotone rescaling, so it changes no ordering and
no merge.
"""
function coupling_distance(J::AbstractMatrix)
    A = abs.(J)
    m = maximum(A)
    m > 0 || error("the interaction matrix is identically zero; nothing to cluster")
    D = 1 .- A ./ m
    D[diagind(D)] .= 0.0
    return D
end

"""
    profile_distance(J) -> Matrix

The second reading: sites are close when they couple to the same partners. Cosine
distance between rows of `|J|`, `1 - <a_i, a_j> / (||a_i|| ||a_j||)`, which is in
`[0, 1]` because the rows are non-negative. Cosine and not correlation, because
subtracting a row mean would make "couples to nothing" a direction rather than the
absence of one; a site whose row is identically zero shares no partner with anybody
and is put at distance 1 from everything, including other empty rows.
"""
function profile_distance(J::AbstractMatrix)
    A = abs.(J)
    n = size(A, 1)
    nrm = [sqrt(sum(abs2, view(A, i, :))) for i in 1:n]
    D = ones(n, n)
    for i in 1:n, j in 1:n
        if nrm[i] > 0 && nrm[j] > 0
            D[i, j] = 1 - dot(view(A, i, :), view(A, j, :)) / (nrm[i] * nrm[j])
        end
    end
    D[diagind(D)] .= 0.0
    return D
end

"""
    linkage_tree(D; linkage) -> (merges, heights, order)

Agglomerative clustering of the `n` objects with dissimilarity `D`. Returns the merge
list in the usual encoding -- leaves are `1:n`, the `m`-th merge creates node `n + m`
and `merges[m, :]` are the two nodes it joins -- the height of each merge, and a leaf
order for drawing.

The three linkages share one Lance-Williams update, which is why they cost the same:
when clusters `i` and `j` join, the distance from the new cluster to every remaining
`k` is a fixed function of `d(i,k)` and `d(j,k)` -- their size-weighted mean
(average), the larger (complete) or the smaller (single). Nothing else about the
merged members is needed, so the whole clustering runs on the distance matrix alone.

`order` is the standard depth-first leaf order with one refinement: at every internal
node the two subtrees, and the direction of each, are chosen so that the two leaves
meeting at the join are as similar as possible. The tree does not fix left from right
-- flipping a subtree is the same tree -- so this is free, and without it adjacent
columns of the drawn matrix are adjacent for no reason.
"""
function linkage_tree(D::AbstractMatrix; linkage::Symbol=:average)
    linkage in (:average, :complete, :single) ||
        error("unknown linkage $linkage; expected :average, :complete or :single")
    n = size(D, 1)
    d = Matrix{Float64}(D)
    for i in 1:n
        d[i, i] = Inf                       # never merge a cluster with itself
    end
    node = collect(1:n)                     # node id currently held by each row
    sz = ones(Int, n)
    alive = trues(n)
    merges = zeros(Int, n - 1, 2)
    heights = zeros(n - 1)

    for m in 1:(n - 1)
        bi, bj, best = 0, 0, Inf
        for i in 1:n, j in (i + 1):n
            (alive[i] && alive[j]) || continue
            d[i, j] < best && ((bi, bj, best) = (i, j, d[i, j]))
        end
        merges[m, :] = [node[bi], node[bj]]
        heights[m] = best
        for k in 1:n
            (alive[k] && k != bi && k != bj) || continue
            d[bi, k] = d[k, bi] =
                linkage === :average ?
                    (sz[bi] * d[bi, k] + sz[bj] * d[bj, k]) / (sz[bi] + sz[bj]) :
                linkage === :complete ? max(d[bi, k], d[bj, k]) :
                                        min(d[bi, k], d[bj, k])
        end
        sz[bi] += sz[bj]
        alive[bj] = false
        node[bi] = n + m
        d[bi, bi] = Inf
    end

    # Leaves under every node, memoized: needed for the ordering, the cut and the
    # branch colours, and cheap enough to keep.
    leaves = Vector{Vector{Int}}(undef, 2n - 1)
    for i in 1:n
        leaves[i] = [i]
    end
    for m in 1:(n - 1)
        leaves[n + m] = vcat(leaves[merges[m, 1]], leaves[merges[m, 2]])
    end

    # Orientation: of the four ways to lay two subtrees end to end, take the one whose
    # meeting leaves are closest in the ORIGINAL dissimilarity.
    function ordered(v)
        v <= n && return [v]
        a = ordered(merges[v - n, 1])
        b = ordered(merges[v - n, 2])
        opts = ((a, b), (reverse(a), b), (a, reverse(b)), (reverse(a), reverse(b)))
        best_opt = argmin([D[last(x), first(y)] for (x, y) in opts])
        x, y = opts[best_opt]
        return vcat(x, y)
    end

    return merges, heights, ordered(2n - 1), leaves
end

"""
    cutree(merges, leaves, n, k) -> Vector{Int}

Labels from cutting the tree into `k` clusters: undo the last `k - 1` merges and take
what is left. Heights increase monotonically under average, complete and single
linkage, so this is exactly the horizontal cut at the height of merge `n - k + 1`.
"""
function cutree(merges, leaves, n::Integer, k::Integer)
    roots = [2n - 1]
    while length(roots) < k
        # split the root formed by the latest merge, i.e. the tallest one left
        splittable = findall(>(n), roots)
        isempty(splittable) && error("cannot cut into $k clusters: only $(length(roots)) left")
        i = splittable[argmax(roots[splittable])]
        v = roots[i]
        deleteat!(roots, i)
        append!(roots, merges[v - n, :])
    end
    labels = zeros(Int, n)
    for (c, r) in enumerate(sort(roots, by = r -> minimum(leaves[r])))
        labels[leaves[r]] .= c
    end
    return labels
end

"""
    adjusted_rand(a, b) -> Float64

Agreement between two partitions, counted over PAIRS of objects and corrected for the
agreement expected from the two label-size distributions alone. 1 is identity, 0 is
what a random partition with those sizes scores. The correction is what makes it
usable here: a lopsided cut agrees with the truth about most pairs for free.
"""
function adjusted_rand(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    n = length(a)
    ua, ub = sort(unique(a)), sort(unique(b))
    N = [count(i -> a[i] == x && b[i] == y, 1:n) for x in ua, y in ub]
    c2(v) = v * (v - 1) / 2
    idx = sum(c2, N)
    ea = sum(c2, sum(N, dims=2))
    eb = sum(c2, sum(N, dims=1))
    expected = ea * eb / c2(n)
    maxi = (ea + eb) / 2
    return (idx - expected) / (maxi - expected)
end

"""
    cophenetic_correlation(D, merges, heights, leaves, n) -> Float64

How faithfully the tree reproduces the distances it was built from: the correlation
between each pair's input dissimilarity and the height at which the tree first puts
the two in one cluster. A hierarchy is a strong claim about a set of distances --
that they are approximately ultrametric -- and this is the one number that says
whether the claim holds before any cut of the tree is interpreted.
"""
function cophenetic_correlation(D, merges, heights, leaves, n)
    C = zeros(n, n)
    for m in 1:(n - 1)
        for i in leaves[merges[m, 1]], j in leaves[merges[m, 2]]
            C[i, j] = C[j, i] = heights[m]
        end
    end
    pairs = [(i, j) for i in 1:n for j in (i + 1):n]
    return cor([D[i, j] for (i, j) in pairs], [C[i, j] for (i, j) in pairs])
end

# ================================================================ the trees ==

"""
    run_test(labels) -> (runs, expected, sd, z)

Runs test on a two-label sequence: a run is a maximal stretch of one label, so few
runs means like sits beside like. Under the null that all arrangements of the two
labels are equally likely, the number of runs has mean `1 + 2 n1 n2 / n` and a known
variance, both exact -- no simulation needed. Applied to the module labels read along
the tree's leaf order, it asks whether the tree grouped same-protein sites together,
which is a different question from whether any cut of it recovers the partition.
"""
function run_test(labels::AbstractVector{<:Integer})
    n = length(labels)
    n1 = count(==(first(sort(unique(labels)))), labels)
    n2 = n - n1
    (n1 == 0 || n2 == 0) && return (1, 1.0, 0.0, 0.0)
    R = 1 + count(i -> labels[i] != labels[i + 1], 1:(n - 1))
    μ = 1 + 2 * n1 * n2 / n
    σ = sqrt(2 * n1 * n2 * (2 * n1 * n2 - n) / (n^2 * (n - 1)))
    return R, μ, σ, (R - μ) / σ
end

"""
    cross_boundary_share(J, boundary; rng, n_shuffles) -> (observed, null_mean, null_sd, z)

Share of the total |J| that sits on pairs crossing the module boundary, against the
share obtained by dealing this matrix's own coupling values out over all pairs at
random. The null is a permutation of PLACEMENT only: same values, same number of
them, nothing about their magnitudes changed, so anything the comparison shows is
about where the couplings are and not how big they are.

With two equal modules almost exactly half of all site pairs cross the boundary, so
the null sits near a half whatever the couplings look like; that is the number the
observed share has to be read against, and it is why this says something no tree is
needed for.
"""
function cross_boundary_share(J::AbstractMatrix, labels::AbstractVector{<:Integer};
                              rng, n_shuffles::Integer=500)
    p = size(J, 1)
    idx = [(i, j) for i in 1:p for j in (i + 1):p]
    vals = [abs(J[i, j]) for (i, j) in idx]
    cross = [labels[i] != labels[j] for (i, j) in idx]
    total = sum(vals)
    observed = sum(vals[cross]) / total
    null = [sum(shuffle(rng, vals)[cross]) / total for _ in 1:n_shuffles]
    return observed, mean(null), std(null), (observed - mean(null)) / std(null)
end

"""
    cluster_and_score(D, true_labels; k_show) -> NamedTuple

One tree, cut two ways: at k = 2 for the test against the true modules, and at
`k_show` for display. The k = 2 cut is the number the module hypothesis names in
advance, which is what makes its adjusted Rand index interpretable — a cut chosen by
scanning k would be an optimistically biased statistic needing its own null.
"""
function cluster_and_score(D::AbstractMatrix, true_labels::AbstractVector; k_show::Integer=4)
    merges, heights, order, leaves = linkage_tree(D)
    n = size(D, 1)
    coph = cophenetic_correlation(D, merges, heights, leaves, n)
    labels2 = cutree(merges, leaves, n, 2)
    labels_show = cutree(merges, leaves, n, k_show)
    return (; merges, heights, order, leaves, labels2, labels_show,
              ari = adjusted_rand(labels2, true_labels), coph)
end

"""
    cluster_matrix(name, J, true_labels, boundary; distance, k_show, rng, n_shuffles)

Everything this analysis claims about one interaction matrix, computed and returned
without printing: the tree under the chosen dissimilarity, the same tree's scores
under the OTHER dissimilarity (so a conclusion that turns on that choice cannot be
stated without noticing), the cross-boundary share against its placement null, and
the runs test on the leaf order.
"""
function cluster_matrix(name, J::AbstractMatrix, true_labels::AbstractVector;
                        distance::Symbol=:coupling,
                        k_show::Integer=4, rng, n_shuffles::Integer=500)
    D = distance === :profile ? profile_distance(J) : coupling_distance(J)
    Dalt = distance === :profile ? coupling_distance(J) : profile_distance(J)
    t = cluster_and_score(D, true_labels; k_show)
    alt = cluster_and_score(Dalt, true_labels; k_show)
    xs = cross_boundary_share(J, true_labels; rng, n_shuffles)
    runs = run_test(true_labels[t.order])
    return (; name, J, D, t.merges, t.heights, t.order, t.leaves,
              t.labels2, t.labels_show, t.ari, t.coph,
              alt_name = distance === :profile ? "coupling" : "profile",
              alt_labels2 = alt.labels2, ari_alt = alt.ari, coph_alt = alt.coph,
              cross_share = xs, runs)
end

"""
    modular_control(J, boundary; rng) -> Matrix

`J`'s own nonzero couplings, dealt out at random among within-module pairs only: same
count, same strengths, same sparsity, modular placement. The positive control — it
says whether this clustering can find a boundary that is really there, in a matrix of
exactly this size and this shape. A control drawn from some other distribution would
confound placement with count, magnitude and sparsity all at once.
"""
function modular_control(J::AbstractMatrix, labels::AbstractVector{<:Integer}; rng)
    p = size(J, 1)
    vals = [abs(J[i, j]) for i in 1:p for j in (i + 1):p if J[i, j] != 0]
    within = [(i, j) for i in 1:p for j in (i + 1):p if labels[i] == labels[j]]
    length(vals) <= length(within) ||
        error("$(length(vals)) nonzero couplings will not fit in $(length(within)) within-module pairs")
    slots = shuffle(rng, within)[1:length(vals)]
    C = zeros(p, p)
    for (v, (i, j)) in zip(shuffle(rng, vals), slots)
        C[i, j] = C[j, i] = v
    end
    return C
end
