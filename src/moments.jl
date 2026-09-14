# Second moments of ancestry, and the small readings taken off a loading vector.
# Shared by the covariance analysis and the ICA that follows it.

"""
    ancestry_moments(ancestry, mask; chunk) -> (μ, C, n)

Mean and covariance of the ancestry over the columns selected by `mask`, accumulated
in chunks so the `n_sites x n_chimeras` matrix is never materialized in Float64 (that
would be ~0.9 GB at 1e6 chimeras).

Everything downstream is computed on this covariance and never on a normalized
correlation: ancestry is a parent label, so the variance is `1/4` at every site by
construction in an unselected library, and normalizing would only rescale the picture
while discarding the natural units — and would additionally hide the variance drift
that selection itself produces.
"""
function ancestry_moments(ancestry::AbstractMatrix, mask::AbstractVector{Bool};
                          chunk::Integer=50_000)
    p = size(ancestry, 1)
    total = zeros(p)
    gram = zeros(p, p)
    n = 0
    buf = Matrix{Float64}(undef, p, chunk)
    for lo in 1:chunk:size(ancestry, 2)
        hi = min(lo + chunk - 1, size(ancestry, 2))
        cols = [j for j in lo:hi if mask[j]]
        isempty(cols) && continue
        X = @view buf[:, 1:length(cols)]
        @views X .= ancestry[:, cols]
        total .+= vec(sum(X, dims=2))
        mul!(gram, X, X', 1.0, 1.0)
        n += length(cols)
    end
    μ = total ./ n
    C = gram ./ n .- μ * μ'
    return μ, C, n
end

"""
    protein1_weight(v, boundary) -> Float64

Fraction of a mode's squared loading falling in the first module. The one-number
summary of where a mode lives; 1 and 0 mean it is confined to one protein, 0.5 that
it straddles.
"""
protein1_weight(v, boundary::Integer) = sum(abs2, v[1:boundary]) / sum(abs2, v)

"""
    protein1_weight(v, labels) -> Float64

The same reading when the modules are not contiguous: the fraction of squared loading on
the sites `labels` marks as module 1. Identical to the boundary form when the labels are
sorted, and the only correct form once the sites have been relabelled.
"""
protein1_weight(v, labels::AbstractVector{<:Integer}) =
    sum(abs2, v[labels .== 1]) / sum(abs2, v)

"""
    loading_span(v; frac) -> (lo, hi)

Smallest window of consecutive sites holding `frac` of a vector's squared loading. A
component that is a module reads as a window ending at the boundary; one that is an
artifact of the crossover process reads as a short window anywhere.
"""
function loading_span(v; frac::Real=0.9)
    w = abs2.(v) ./ sum(abs2, v)
    best = (1, length(w))
    for i in eachindex(w)
        acc = 0.0
        for j in i:length(w)
            acc += w[j]
            if acc >= frac
                (j - i) < (best[2] - best[1]) && (best = (i, j))
                break
            end
        end
    end
    return best
end

"""
    ar1_precision_check(C; ρ, v) -> NamedTuple

Compare the pseudo-inverse of an ancestry covariance against the closed form for a
two-state Markov chain: `C⁺` tridiagonal with interior diagonal `(1+ρ²)/(1-ρ²)/v`,
corners `1/(1-ρ²)/v` and first off-diagonal `-ρ/(1-ρ²)/v`. This is the check that the
crossover generator is what it claims to be.

It belongs on the UNCONDITIONED library and nowhere else. Dropping the intact parents
is not a Markov operation, so a screened library stays tridiagonal but at an effective
ρ slightly below `1 - 2p`, and the exact values no longer apply.
"""
function ar1_precision_check(C::AbstractMatrix; ρ::Real, v::Real=0.25)
    P = pinv(C)
    off_band = maximum(abs(P[i, j]) for i in axes(P, 1), j in axes(P, 2) if abs(i - j) >= 2)
    return (; interior = median(diag(P)[2:(end - 1)]),
              interior_theory = (1 + ρ^2) / (1 - ρ^2) / v,
              corners = (P[1, 1] + P[end, end]) / 2,
              corners_theory = 1 / (1 - ρ^2) / v,
              offdiag = median(diag(P, 1)),
              offdiag_theory = -ρ / (1 - ρ^2) / v,
              max_beyond_band = off_band)
end
