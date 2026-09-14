# Supervised machinery: metrics, the linear designs and their penalized solvers, and
# the neural models. Moved out of the analysis scripts because every experiment fits
# the same models to its own library.
#
# Inputs are spin vectors, `n_sites x n` with entries in {-1,+1}. Nothing here assumes
# anything about
# how sites relate to one another: no ordering, no neighbourhood, no lattice. A real
# shuffling experiment would not know the structure, so neither do these models —
# which is also what rules out a convolutional model.

# ============================================================== metrics =====

"""Fraction of variance explained. Negative means worse than predicting the mean."""
r2(ŷ, y) = 1 - sum(abs2, y .- ŷ) / sum(abs2, y .- mean(y))

"""
    auroc(score, label) -> Float64

Area under the ROC curve, computed from rank statistics (ties handled), so it
needs no threshold sweep and no extra dependency.
"""
function auroc(score::AbstractVector, label::AbstractVector{Bool})
    n_pos = count(label)
    n_neg = length(label) - n_pos
    (n_pos == 0 || n_neg == 0) && return NaN
    r = tiedrank(score)
    return (sum(r[label]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end

"""Average precision: area under the precision-recall curve, the useful summary
when the classes are unbalanced."""
function auprc(score::AbstractVector, label::AbstractVector{Bool})
    o = sortperm(score, rev=true)
    tp = 0
    total_pos = count(label)
    total_pos == 0 && return NaN
    ap = 0.0
    for (k, i) in enumerate(o)
        if label[i]
            tp += 1
            ap += tp / k
        end
    end
    return ap / total_pos
end

"""
    evaluate(ŷ, y, threshold) -> NamedTuple

Regression and classification quality in one pass. Fitness here is a spike near the
ceiling with a long tail downward, so a single R2 hides which regime is being fit:
`r2_functional` and `r2_broken` split it. Classification treats "functional" as the
positive class and is derived by thresholding the predicted fitness.
"""
function evaluate(ŷ::AbstractVector, y::AbstractVector, threshold::Real)
    lab = y .>= threshold
    hi = findall(lab)
    lo = findall(.!lab)
    return (; r2 = r2(ŷ, y),
              rmse = sqrt(mean(abs2, ŷ .- y)),
              spearman = corspearman(ŷ, y),
              r2_functional = isempty(hi) ? NaN : r2(ŷ[hi], y[hi]),
              r2_broken = isempty(lo) ? NaN : r2(ŷ[lo], y[lo]),
              auroc = auroc(ŷ, lab),
              auprc = auprc(ŷ, lab),
              n = length(y))
end

function print_metrics(name, m)
    @printf("  %-22s R2 %+.4f  rmse %.4f  rho %+.3f  AUROC %.4f  AUPRC %.4f  (n=%d)\n",
            name, m.r2, m.rmse, m.spearman, m.auroc, m.auprc, m.n)
    flush(stdout)
end

# ====================================================== closed-form models ==

"""
    design_additive(X) -> Matrix{Float64}

One column per site plus an intercept: `n x (p+1)`. Constant sites are kept. They
are collinear with the intercept, which ridge handles -- nothing is deleted, so
feature index and site index stay identical.
"""
design_additive(X::AbstractMatrix) = [ones(size(X, 2)) Float64.(permutedims(X))]

"""
    pair_index(p) -> (pairs, npairs)

All unordered site pairs `(i,j), i<j`, in a fixed order shared by every routine
that builds or interprets a pairwise model.
"""
function pair_index(p::Integer)
    pairs = Tuple{Int,Int}[]
    for i in 1:p, j in (i + 1):p
        push!(pairs, (i, j))
    end
    return pairs
end

"""
    design_pairwise!(Φ, X, cols, pairs)

Fill `Φ` (`(1+p+npairs) x length(cols)`) with intercept, site values, and all
pairwise products, for the columns `cols` of `X`. Written in place so the full
design matrix -- 6106 x 200000 here, 9.8 GB in Float64 -- is never materialized.
"""
function design_pairwise!(Φ::AbstractMatrix, X::AbstractMatrix, cols, pairs)
    p = size(X, 1)
    @inbounds for (c, j) in enumerate(cols)
        Φ[1, c] = 1.0
        for i in 1:p
            Φ[1 + i, c] = X[i, j]
        end
        for (k, (a, b)) in enumerate(pairs)
            Φ[1 + p + k, c] = X[a, j] * X[b, j]
        end
    end
    return Φ
end

"""
    gram_pairwise(X, y, cols, pairs; chunk) -> (G, b)

Accumulate `Φ'Φ` and `Φ'y` by streaming over chunks of samples. Same trick as
`ancestry_moments` in 05_pca.jl: the Gram is small and fixed while the design
matrix is enormous, so only the Gram is kept.

The Gram depends only on the data, not on any model choice, so one accumulation
serves every fit made on the same rows -- ridge, and the whole L1 penalty path.
"""
function gram_pairwise(X::AbstractMatrix, y::AbstractVector, cols, pairs; chunk::Int=2000)
    m = 1 + size(X, 1) + length(pairs)
    G = zeros(m, m)
    b = zeros(m)
    Φ = Matrix{Float64}(undef, m, chunk)
    for lo in 1:chunk:length(cols)
        hi = min(lo + chunk - 1, length(cols))
        sub = @view cols[lo:hi]
        Φc = @view Φ[:, 1:length(sub)]
        design_pairwise!(Φc, X, sub, pairs)
        mul!(G, Φc, Φc', 1.0, 1.0)
        mul!(b, Φc, @view(y[sub]), 1.0, 1.0)
    end
    return G, b
end

"""
    ridge_solve(G, b, λ) -> β

Solve `(G + λI) β = b` from the accumulated Gram, so a fit costs a Cholesky on the
Gram rather than a pass over the design matrix.

λ is never applied to the intercept.
"""
function ridge_solve(G::AbstractMatrix, b::AbstractVector, λ::Real)
    A = G + λ * I
    A[1, 1] -= λ                           # leave the intercept unpenalized
    return cholesky!(Symmetric(A)) \ b
end

"""
    lasso_lambda_max(G, b; unpenalized) -> Float64

The smallest penalty at which every penalized coefficient is exactly zero, in closed
form from the Gram. At `β = 0` except for the intercept, coordinate `j`'s
subgradient condition is satisfied at zero exactly when `|ρ_j| <= λ`, so the largest
`|ρ_j|` is the threshold above which nothing enters the model.

This is what makes a penalty comparable across designs and across experiments: `λ`
itself carries the sample count and the target scale, `λ / λ_max` carries neither.
"""
function lasso_lambda_max(G::AbstractMatrix, b::AbstractVector; unpenalized::Integer=1)
    β0 = G[unpenalized, unpenalized] > 0 ? b[unpenalized] / G[unpenalized, unpenalized] : 0.0
    return maximum(abs(b[j] - β0 * G[j, unpenalized])
                   for j in eachindex(b) if j != unpenalized)
end

"""
    lasso_core!(β, G, b, λ; max_sweeps, tol) -> β

Cyclic coordinate descent for `½ β'Gβ − b'β + λ‖β‖₁`, every coordinate penalized. `G`
and `b` must already be CENTRED (see `center_gram`); `β` is updated in place and warm
starts are simply a `β` that is already close.

There is no closed form for L1, but the coordinate update has one --

    β_j  <-  S(b_j − Σ_{k≠j} G_jk β_k,  λ) / G_jj,    S(z,λ) = sign(z)·max(|z|−λ, 0)

-- so a fit costs sweeps over the Gram rather than passes over the data, exactly like
`ridge_solve`. The running residual `r = b − Gβ` makes each update O(1) to propose and
O(m) to accept, and because most coefficients stay at zero the sweeps are cheap: after
each full sweep the loop cycles on the active set alone until it settles, then re-checks
everything.

Stopping is on the OPTIMALITY CONDITION itself, not on how far the coefficients last
moved. At the optimum the residual satisfies `r_j = λ·sign(β_j)` on the support and
`|r_j| ≤ λ` off it, so the largest excursion outside that band certifies exactly how
suboptimal the current `β` is -- and dividing it by λ makes `tol` mean the same thing at
every point of a path, which a coefficient-change threshold does not.

Identically-zero features (`G_jj = 0`: the constant sites Step 6 keeps, and every pair
involving one) carry no information and stay at zero. Where ridge needs `λI` to make
them well-posed, L1 handles them by construction.
"""
function lasso_core!(β::AbstractVector, G::AbstractMatrix, b::AbstractVector, λ::Real;
                     max_sweeps::Integer=1000, tol::Real=1e-4)
    m = size(G, 1)
    r = b - G * β
    soft(z, t) = z > t ? z - t : (z < -t ? z + t : zero(z))

    # Keeps `r` consistent with `β`.
    function update!(j)
        Gjj = G[j, j]
        Gjj > 0 || return
        ρ = r[j] + Gjj * β[j]
        βnew = soft(ρ, λ) / Gjj
        δ = βnew - β[j]
        δ == 0 && return
        @views r .-= δ .* G[:, j]
        β[j] = βnew
        return
    end

    function kkt_violation(idxs)
        v = 0.0
        for j in idxs
            G[j, j] > 0 || continue
            vj = β[j] != 0 ? abs(r[j] - λ * sign(β[j])) : max(0.0, abs(r[j]) - λ)
            v = max(v, vj)
        end
        return v
    end

    # `max_sweeps` is a budget over ALL sweeps, inner and outer. Counting it per loop
    # instead would make the worst case the product of the two, which at the dense end
    # of a path is the difference between seconds and hours.
    goal = tol * λ
    sweeps = 0
    violation = Inf
    while sweeps < max_sweeps
        for j in 1:m
            update!(j)
        end
        sweeps += 1
        # `r` is carried incrementally through millions of rank-one updates, so it drifts
        # from `b − Gβ` by accumulated rounding, and the stopping test reads `r` directly.
        # Refresh it exactly before testing: one matrix-vector product against a sweep
        # that already costs many.
        r .= b .- G * β
        violation = kkt_violation(1:m)
        violation <= goal && break
        # The full sweep above is the only place a coefficient can enter; everything
        # after it is refinement, and refinement only ever touches the active set.
        active = findall(!iszero, β)
        while sweeps < max_sweeps
            for j in active
                update!(j)
            end
            sweeps += 1
            kkt_violation(active) <= goal && break
        end
    end
    violation <= goal ||
        @warn "lasso_core! hit its sweep budget; KKT violation is above the target" λ max_sweeps relative_violation=violation/λ
    return β
end

"""
    lasso_solve(G, b, λ; β0, max_sweeps, tol, unpenalized) -> β

`lasso_core!` on the centred system, with the unpenalized coordinate profiled out and
put back afterwards. Takes and returns a full-length `β` indexed like `G`, so it is a
drop-in counterpart to `ridge_solve` on the same `G` and `b`.
"""
function lasso_solve(G::AbstractMatrix, b::AbstractVector, λ::Real;
                     β0=nothing, max_sweeps::Integer=1000, tol::Real=1e-4,
                     unpenalized::Integer=1)
    G̃, b̃, g, n = center_gram(G, b; unpenalized)
    keep = [j for j in axes(G, 1) if j != unpenalized]
    βpen = β0 === nothing ? zeros(length(keep)) : collect(float.(β0[keep]))
    lasso_core!(βpen, G̃, b̃, λ; max_sweeps, tol)
    β = zeros(size(G, 1))
    β[keep] = βpen
    β[unpenalized] = (b[unpenalized] - dot(g, βpen)) / n
    return β
end

"""
    center_gram(G, b; unpenalized) -> (G̃, b̃, counts, n)

Recentre a Gram so the penalized features have mean zero, returning the reduced system
with the unpenalized column removed.

With an unpenalized intercept, profiling it out of the least-squares objective leaves
exactly the same penalized problem on centred data, so this changes nothing about the
solution -- it is a preconditioner, not a different model. A column with a nonzero mean
is correlated with the intercept, and cyclic coordinate descent converges very slowly in
that geometry: measured under the earlier 0/1 coding, where every column had a mean near
0.5, the uncentred problem had not settled its support after 60000 sweeps while the
centred one converged in tens. Spin coding removes most of that offset by itself, since
a varying site now has a mean near zero, so centring should matter far less than it did.
It is kept because it is free, exactly equivalent, and still corrects the constant sites,
whose columns are all one value.

Everything needed is already in the Gram: with `g` the unpenalized row (the column sums)
and `n = G[1,1]` the sample count, `X̃X̃' = G − gg'/n` and `X̃ỹ = b − g·(b₁/n)`.
"""
function center_gram(G::AbstractMatrix, b::AbstractVector; unpenalized::Integer=1)
    keep = [j for j in axes(G, 1) if j != unpenalized]
    n = G[unpenalized, unpenalized]
    g = G[unpenalized, keep]
    ȳ = b[unpenalized] / n
    G̃ = G[keep, keep] .- (g * g') ./ n
    b̃ = b[keep] .- ȳ .* g
    return G̃, b̃, g, n
end

"""
    lasso_path(G, b, λs; unpenalized, kwargs...) -> Vector{Vector{Float64}}

`lasso_solve` at each λ in turn, each fit warm-started from the previous one. `λs` must
be DESCENDING: starting at the sparse end and relaxing means every fit begins near its
own solution, which is what makes a whole path cost little more than one fit at the
smallest λ.

The Gram is centred ONCE for the whole path rather than per fit -- centring is an O(m²)
transformation and the path would otherwise pay for it at every λ.
"""
function lasso_path(G::AbstractMatrix, b::AbstractVector, λs::AbstractVector;
                    unpenalized::Integer=1, kwargs...)
    issorted(λs, rev=true) || error("lasso_path needs descending λs for warm starts")
    G̃, b̃, g, n = center_gram(G, b; unpenalized)
    keep = [j for j in axes(G, 1) if j != unpenalized]
    βpen = zeros(length(keep))
    out = Vector{Vector{Float64}}(undef, length(λs))
    for (k, λ) in enumerate(λs)
        lasso_core!(βpen, G̃, b̃, λ; kwargs...)
        β = zeros(size(G, 1))
        β[keep] = βpen
        β[unpenalized] = (b[unpenalized] - dot(g, βpen)) / n
        out[k] = β
    end
    return out
end

"""Predict from a pairwise coefficient vector without building the design matrix."""
function predict_pairwise(β::AbstractVector, X::AbstractMatrix, cols, pairs;
                          chunk::Int=2000)
    m = 1 + size(X, 1) + length(pairs)
    ŷ = Vector{Float64}(undef, length(cols))
    Φ = Matrix{Float64}(undef, m, chunk)
    for lo in 1:chunk:length(cols)
        hi = min(lo + chunk - 1, length(cols))
        sub = @view cols[lo:hi]
        Φc = @view Φ[:, 1:length(sub)]
        design_pairwise!(Φc, X, sub, pairs)
        mul!(@view(ŷ[lo:hi]), Φc', @view(β[:]))
    end
    return ŷ
end

# ========================================================= neural models ====

"""
    mlp(p, hidden; dropout) -> Chain

A plain fully-connected network on the raw spin vector. Every site connects to every
hidden unit, so the model is free to discover any interaction structure and is told
nothing about which sites are near which.
"""
mlp(p::Integer, hidden::AbstractVector{<:Integer}; dropout::Real=0.1) =
    Chain(mlp_body(p, hidden; dropout), Dense(hidden[end] => 1))

"""Hidden stack of `mlp`, without the output layer."""
function mlp_body(p::Integer, hidden::AbstractVector{<:Integer}; dropout::Real=0.1)
    layers = Any[]
    d = p
    for h in hidden
        push!(layers, Dense(d => h, relu))
        dropout > 0 && push!(layers, Dropout(Float32(dropout)))
        d = h
    end
    return Chain(layers...)
end

"""
    SiteAttention

NOT USED by the pipeline. Kept because it works and may be worth revisiting on a
smaller system or with a GPU: measured on this problem it cost ~110x the MLP's
training time (86 min against 47 s) for a worse score, since 110 tokens attending
to each other dominates everything else at this size. Its validation loss was still
falling when the budget ran out, so that score was a lower bound, not a ceiling.

Self-attention over sites. Each site becomes a token: a learned per-site embedding
(so the model can tell sites apart) plus a learned embedding of the bit it carries.
Attention then lets any site interact with any other, weighted by what the data
supports -- there is no locality prior and no assumed ordering, which is the point.
Tokens are mean-pooled and passed to a small head.
"""
struct SiteAttention{S,V,B,A,N,H}
    site_emb::S      # d x p, learned identity of each site
    val_emb::V       # d x 2, learned embedding of bit value
    bias::B
    attn::A
    norm::N
    head::H
end
Flux.@layer SiteAttention

function SiteAttention(p::Integer; d::Integer=32, nheads::Integer=4, nlayers::Integer=2)
    attn = Tuple(Flux.MultiHeadAttention(d; nheads) for _ in 1:nlayers)
    norm = Tuple(LayerNorm(d) for _ in 1:nlayers)
    return SiteAttention(randn(Float32, d, p) .* 0.05f0,
                         randn(Float32, d, 2) .* 0.05f0,
                         zeros(Float32, d),
                         attn, norm,
                         Chain(Dense(d => d, relu), Dense(d => 1)))
end

function (m::SiteAttention)(x::AbstractMatrix)     # x: p x batch, values in {0,1}
    d, p = size(m.site_emb)
    b = size(x, 2)
    # token = site identity + value embedding, broadcast over the batch
    v = m.val_emb[:, 1] .+ (m.val_emb[:, 2] .- m.val_emb[:, 1]) .* reshape(x, 1, p, b)
    h = reshape(m.site_emb, d, p, 1) .+ v .+ m.bias
    for (attn, norm) in zip(m.attn, m.norm)
        y, _ = attn(h)
        h = norm(h .+ y)                            # residual + normalize
    end
    return m.head(dropdims(mean(h, dims=2), dims=2))
end

"""
    train_net!(model, Xtr, ytr, Xva, yva; ...) -> (model, history)

Adam with early stopping on validation loss, keeping the best parameters seen. Data
is `features x samples`; targets are standardized by the caller.
"""
function train_net!(model, Xtr, ytr, Xva, yva;
                    rng::AbstractRNG, epochs::Int=60, batch::Int=512,
                    lr::Real=1e-3, patience::Int=8, label::AbstractString="net")
    opt = Flux.setup(Adam(Float32(lr)), model)
    n = size(Xtr, 2)
    best = (loss = Inf, state = deepcopy(Flux.state(model)), epoch = 0)
    history = Float64[]
    val_loss() = (Flux.testmode!(model);
                  l = mean(abs2, vec(model(Xva)) .- yva);
                  Flux.trainmode!(model); l)
    for epoch in 1:epochs
        order = shuffle(rng, 1:n)
        Flux.trainmode!(model)
        for lo in 1:batch:n
            idx = @view order[lo:min(lo + batch - 1, n)]
            xb = Xtr[:, idx]
            yb = ytr[idx]
            g = Flux.gradient(mm -> mean(abs2, vec(mm(xb)) .- yb), model)[1]
            Flux.update!(opt, model, g)
        end
        l = val_loss()
        push!(history, l)
        if l < best.loss
            best = (loss = l, state = deepcopy(Flux.state(model)), epoch = epoch)
        elseif epoch - best.epoch >= patience
            @printf("    %s: early stop at epoch %d (best %d, val %.5f)\n",
                    label, epoch, best.epoch, best.loss)
            break
        end
        epoch % 5 == 0 && @printf("    %s: epoch %3d  val %.5f\n", label, epoch, l)
        # stdout is block-buffered when redirected to a file, so a long training run
        # would otherwise show nothing at all until it finished
        flush(stdout)
    end
    Flux.loadmodel!(model, best.state)
    Flux.testmode!(model)
    return model, history
end

"""Predict in batches, so a large evaluation set never becomes one huge forward pass."""
function predict_net(model, X; batch::Int=8192)
    Flux.testmode!(model)
    out = Vector{Float32}(undef, size(X, 2))
    for lo in 1:batch:size(X, 2)
        hi = min(lo + batch - 1, size(X, 2))
        out[lo:hi] .= vec(model(@view X[:, lo:hi]))
    end
    return out
end

# ================================================== fitting the whole family ====

"""
    model_hyperparameters(; kwargs...) -> NamedTuple

The knobs `fit_models` needs, in one object so an experiment's parameter block and the
fit stay in one place. Every field is required; there are no defaults here, because a
default hyperparameter that nobody chose is exactly the thing a parameter sweep must
not have.
"""
model_hyperparameters(; ridge_lambda, lasso_n_lambda, lasso_lambda_min_ratio,
                      lasso_max_sweeps, lasso_refit_sweeps, lasso_tol, lasso_path_tol,
                      mlp_hidden, n_epochs, batch_size, learning_rate,
                      early_stop_patience, train_seed) =
    (; ridge_lambda, lasso_n_lambda, lasso_lambda_min_ratio, lasso_max_sweeps,
       lasso_refit_sweeps, lasso_tol, lasso_path_tol, mlp_hidden, n_epochs, batch_size,
       learning_rate, early_stop_patience, train_seed)

const MODEL_NAMES = ["additive", "additive_lasso", "pairwise", "pairwise_lasso", "mlp"]

"""
    fit_models(X, y, threshold, train_ids, val_ids, test_ids, hp;
               tag, artifacts) -> (metrics, extra)

Fit every model on `train_ids` and score them on `test_ids`. Targets are standardized
using training statistics only and every prediction is mapped back before any metric is
computed, so all numbers are in fitness units.

The L1 fits reuse the L2 fits' Gram and differ from them in nothing but the penalty, so
the pair is a controlled comparison rather than two models. The reason to want L1 at all
is the interaction matrix: a modular system has EXACTLY ZERO cross-boundary coupling,
and L2 shrinks toward zero but never to it, so every pair comes back nonzero and "is
this pair coupled?" could only be answered by a threshold chosen after the fit.

Their penalty is expressed as a fraction of `λ_max` and chosen on `val_ids`, which the
linear models otherwise never use; the chosen ratios come back in `extra`.

`artifacts=true` additionally returns the fitted coefficients, the pairwise model's
fields and both its interaction matrices, and the trained network. The network is fitted
to predict fitness and nothing is read off it about which sites matter.
"""
function fit_models(X::AbstractMatrix, y::AbstractVector, threshold::Real,
                    train_ids, val_ids, test_ids, hp;
                    tag::AbstractString, artifacts::Bool=false,
                    Xf::AbstractMatrix=Matrix{Float32}(X))
    p = size(X, 1)
    pairs = pair_index(p)
    μy = mean(y[train_ids]); σy = std(y[train_ids])
    ystd = (y .- μy) ./ σy
    unstd(v) = v .* σy .+ μy
    metrics = Dict{String,Any}()
    extra = Dict{String,Any}()
    chosen_ratios = Dict{String,Float64}()

    score!(name, ŷ) = begin
        m = evaluate(ŷ, y[test_ids], threshold)
        metrics[name] = m
        print_metrics(name, m)
        m
    end

    println("### $tag: train $(length(train_ids)), test $(length(test_ids))")

    # Fit the L1 path from an already-accumulated Gram and return the coefficients at
    # the selected penalty. `predict_std` maps a coefficient vector and a set of ids to
    # STANDARDIZED predictions, so the selection loss is in the units the fit minimizes.
    function fit_lasso(name, G, b, predict_std)
        λmax = lasso_lambda_max(G, b)
        ratios = exp.(range(0, log(hp.lasso_lambda_min_ratio), length=hp.lasso_n_lambda))
        # The path and the refit have different jobs and get different tolerances. The
        # path exists only to RANK penalties on the validation split, and that ranking is
        # insensitive to how tightly each point is solved -- measured on this Gram, the
        # validation R2 at the selected penalty is identical to five decimals at 1e-3,
        # 1e-2 and 3e-2, while the path cost falls 4x. The refit below is the fit that is
        # actually reported, and it keeps the tight tolerance.
        t = @elapsed βs = lasso_path(G, b, λmax .* ratios;
                                     max_sweeps=hp.lasso_max_sweeps, tol=hp.lasso_path_tol)
        # The whole path is printed, not just the winner: the sparsity/accuracy
        # tradeoff is the object of interest and one selected point hides it.
        v0 = var(ystd[val_ids])
        losses = [mean(abs2, predict_std(β, val_ids) .- ystd[val_ids]) for β in βs]
        @printf("  %s path (validation):\n", name)
        for (i, β) in enumerate(βs)
            @printf("    λ/λmax %9.3g   nonzero %5d   val R2 %+.5f%s\n",
                    ratios[i], count(!iszero, β[2:end]), 1 - losses[i] / v0,
                    i == argmin(losses) ? "  <- selected" : "")
        end
        k = argmin(losses)
        # Refit the selected penalty to tolerance, warm-started from the path: the path
        # may be loose, this must not be.
        t2 = @elapsed β = lasso_solve(G, b, λmax * ratios[k]; β0=βs[k],
                                      max_sweeps=hp.lasso_refit_sweeps, tol=hp.lasso_tol)
        chosen_ratios[name] = ratios[k]
        nz = count(!iszero, β[2:end])
        @printf("  (%s path %.1f s + refit %.1f s, %d λ; λ/λmax = %.4g chosen on validation, %d of %d coefficients nonzero)\n",
                name, t, t2, length(ratios), ratios[k], nz, length(β) - 1)
        k == 1 &&
            @warn "$name selected the sparsest λ on the grid; raise lasso_lambda_min_ratio" name
        k == length(ratios) &&
            @warn "$name selected the densest λ on the grid; lower lasso_lambda_min_ratio" name
        return β
    end

    # --- additive ---
    Φtr = design_additive(@view X[:, train_ids])
    G_add, b_add = Φtr'Φtr, Φtr' * ystd[train_ids]
    β_add = ridge_solve(G_add, b_add, hp.ridge_lambda)
    score!("additive", unstd(design_additive(@view X[:, test_ids]) * β_add))

    predict_add(β, ids) = design_additive(@view X[:, ids]) * β
    β_add_l1 = fit_lasso("additive_lasso", G_add, b_add, predict_add)
    score!("additive_lasso", unstd(predict_add(β_add_l1, test_ids)))

    # --- pairwise ---
    # The Gram is small and fixed while the design matrix is enormous, so only the Gram
    # is accumulated; one accumulation serves the ridge fit and the whole L1 path.
    t = @elapsed G, b = gram_pairwise(X, ystd, train_ids, pairs)
    β_pair = ridge_solve(G, b, hp.ridge_lambda)
    @printf("  (pairwise Gram %.1f s, %d features)\n", t, length(β_pair))
    score!("pairwise", unstd(predict_pairwise(β_pair, X, test_ids, pairs)))

    predict_pair(β, ids) = predict_pairwise(β, X, ids, pairs)
    β_pair_l1 = fit_lasso("pairwise_lasso", G, b, predict_pair)
    score!("pairwise_lasso", unstd(predict_pair(β_pair_l1, test_ids)))

    # --- neural ---
    Xtr = Xf[:, train_ids]; ytr = Float32.(ystd[train_ids])
    Xva = Xf[:, val_ids];   yva = Float32.(ystd[val_ids])
    nets = Dict{String,Any}(); histories = Dict{String,Vector{Float64}}()
    Random.seed!(hp.train_seed)            # Flux initializes from the global RNG
    net = mlp(p, hp.mlp_hidden)
    t = @elapsed net, hist = train_net!(net, Xtr, ytr, Xva, yva;
                                        rng=Xoshiro(hp.train_seed), epochs=hp.n_epochs,
                                        batch=hp.batch_size, lr=hp.learning_rate,
                                        patience=hp.early_stop_patience, label="$tag/mlp")
    @printf("  (mlp trained %.1f s, %d epochs, %d params)\n",
            t, length(hist), sum(length, Flux.trainables(net)))
    nets["mlp"] = net; histories["mlp"] = hist
    score!("mlp", unstd(Float64.(predict_net(net, @view Xf[:, test_ids]))))

    if artifacts
        # Same map from coefficients to a site-by-site matrix for both penalties, so the
        # two are read on one scale and any difference between them is the penalty.
        interaction_matrix(β) = begin
            M = zeros(p, p)
            for (k, (a, b_)) in enumerate(pairs)
                M[a, b_] = M[b_, a] = β[1 + p + k] * σy
            end
            M
        end
        # The pairwise model's site terms are FIELDS in the Ising sense, on the same
        # scale as the couplings, so they are kept signed and unscaled rather than
        # reduced to an importance magnitude.
        field_vector(β) = β[2:(p + 1)] .* σy
        extra["beta_additive"] = β_add
        extra["beta_pairwise"] = β_pair
        extra["beta_additive_lasso"] = β_add_l1
        extra["beta_pairwise_lasso"] = β_pair_l1
        extra["J"] = interaction_matrix(β_pair)
        extra["J_lasso"] = interaction_matrix(β_pair_l1)
        extra["h"] = field_vector(β_pair)
        extra["h_lasso"] = field_vector(β_pair_l1)
        extra["nets"] = nets
        extra["histories"] = histories
        extra["mu"] = μy
        extra["sigma"] = σy
        extra["unstd"] = unstd
    end
    extra["lasso_ratios"] = chosen_ratios
    println()
    return metrics, extra
end
