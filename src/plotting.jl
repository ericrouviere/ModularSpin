# Every figure the pipeline produces, one function each.
#
# These live in the package rather than in an experiment's script for the same reason
# the analysis does: a parameter sweep compares experiments, and two experiments drawn
# differently are not comparable. Each function takes computed results and returns a
# figure; nothing here reads or writes data, and the caller decides where a figure goes.
#
# The matplotlib backend must be pinned before PyPlot loads, which is the caller's job
# (`ENV["MPLBACKEND"] = "Agg"` before `using ModularSpin`) — otherwise macOS opens
# windows and a headless Linux box does not, and the same script behaves differently on
# the two machines.

"""
    save_figure(fig, path) -> path

Write a figure and close it, so a long run does not accumulate open figures.
"""
function save_figure(fig, path::AbstractString)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    close(fig)
    println("  wrote $path")
    return path
end

# Module 1 / module 2, used by every marker and strip in this file.
const PROTEIN_COLORS = ["#d95f02", "#1b9e77"]
const CLUSTER_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple",
                        "tab:orange", "tab:brown"]

"""Draw the module boundary on a site-vs-site heatmap."""
function mark_boundary!(ax, boundary::Integer)
    ax.axhline(boundary + 0.5, color="tab:red", lw=1.0)
    ax.axvline(boundary + 0.5, color="tab:red", lw=1.0)
end

"""
    mark_modules!(ax, labels)

Mark which module each row and column belongs to. With contiguous modules that is one
line per axis, as before. Once the sites have been relabelled there is no line to draw —
the modules interleave — so the labels go into two thin strips of their OWN axes, one
below the matrix and one to its left.

Separate axes rather than markers plotted just outside the data limits: a mark drawn in
the matrix's own coordinates, however far out, reads as a bright row and column of the
matrix itself, which is exactly the misreading this is meant to prevent.
"""
function mark_modules!(ax, labels::AbstractVector{<:Integer})
    if issorted(labels)
        mark_boundary!(ax, count(==(first(labels)), labels))
        return nothing
    end
    n = length(labels)
    cmap = PyPlot.matplotlib.colors.ListedColormap(PROTEIN_COLORS)
    below = ax.inset_axes([0, -0.055, 1, 0.03], transform=ax.transAxes)
    below.imshow(reshape(Float64.(labels), 1, n), aspect="auto", interpolation="nearest",
                 cmap=cmap, vmin=0.5, vmax=2.5)
    left = ax.inset_axes([-0.055, 0, 0.03, 1], transform=ax.transAxes)
    left.imshow(reshape(Float64.(labels), n, 1), aspect="auto", interpolation="nearest",
                origin="lower", cmap=cmap, vmin=0.5, vmax=2.5)
    for a in (below, left)
        a.set_xticks([]); a.set_yticks([])
    end
    return nothing
end

"""A symmetric-scale site-by-site heatmap with the boundary marked."""
function heat!(ax, M, title, labels; cmap="RdBu_r", sym=true, lim=nothing)
    lim = something(lim, sym ? maximum(abs, M .- Diagonal(M)) : maximum(abs, M))
    im = ax.imshow(M, cmap=cmap, origin="lower", interpolation="nearest",
                   vmin=-lim, vmax=lim)
    labels isa Integer ? mark_boundary!(ax, labels) : mark_modules!(ax, labels)
    ax.set_title(title, fontsize=9)
    return im
end

# Loading sign is drawn in orange/green rather than SpinModel's red/blue. The palette
# has to be set here because `plotMagnetization!` does not forward a colour dict to
# `see_J!`, and SpinModel is read-only — so this calls the exported `see_J!` itself,
# doing exactly what `plotMagnetization!` does otherwise: the magnetization goes in
# under key -1 (the lattice sites), keys 0 and 1 carry zero couplings, and
# `color_couplings=false` greys the bonds out.
const LOADING_COLORS = Dict(true => "#d95f02",     # positive loading
                            false => "#1b9e77")    # negative loading

"""As `plotMagnetization!`, but with the sign colours under our control."""
function plot_loading!(ax, M::AbstractMatrix; size_point=1, col_dict=LOADING_COLORS)
    Wm, Lm = size(M, 1), size(M, 2) - 1
    see_J!(ax, Dict(-1 => size_point .* M, 0 => zeros(Wm, Lm), 1 => zeros(Wm, Lm));
           color_couplings=false, col_dict)
    return nothing
end

"""
    loading_on_protein(v, p, protein_len, W, L) -> Matrix

Reshape protein `p`'s block of a loading vector into the `W x (L+1)` matrix `see_J!`
expects. Sequence index k within a protein is `W*(layer-1) + i`, which is exactly
column-major order, so `reshape` is the right map.
"""
loading_on_protein(v, p::Integer, protein_len::Integer, W::Integer, L::Integer) =
    reshape(v[((p - 1) * protein_len + 1):(p * protein_len)], W, L + 1)

"""
    mark_module_axis!(ax, labels)

The one-dimensional form of `mark_modules!`, for a bar chart over sites: a line at the
boundary when the modules are contiguous, ticks under every module-2 site when they are
not.
"""
function mark_module_axis!(ax, labels::AbstractVector{<:Integer})
    if issorted(labels)
        ax.axvline(count(==(first(labels)), labels) + 0.5, color="k", lw=1.2)
        return nothing
    end
    n = length(labels)
    below = ax.inset_axes([0, -0.10, 1, 0.05], transform=ax.transAxes)
    below.imshow(reshape(Float64.(labels), 1, n), aspect="auto", interpolation="nearest",
                 cmap=PyPlot.matplotlib.colors.ListedColormap(PROTEIN_COLORS),
                 vmin=0.5, vmax=2.5)
    below.set_xticks([]); below.set_yticks([])
    return nothing
end

# ============================================================ steps 1 - 4 ====

"""Random-system fitness: the unselected reference distribution, and its modules."""
function plot_random_fitness(fits, modules; assay_name="AndGate")
    n_proteins_, n = size(modules)
    fig, axs = subplots(1, 2, figsize=(10, 3.6))

    ax = axs[1]
    ax.hist(fits, bins=40, color="0.4", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("system fitness")
    ax.set_ylabel("count")
    ax.set_title("$assay_name, $n random systems", fontsize=10)

    ax = axs[2]
    for i in 1:n_proteins_
        ax.hist(modules[i, :], bins=40, alpha=0.55, label="protein $i", edgecolor="none")
    end
    ax.set_xlabel("per-module binding,  " * L"E_s - E_b")
    ax.set_ylabel("count")
    ax.set_title("per-module distributions", fontsize=10)
    ax.legend(fontsize=7, frameon=false)

    fig.tight_layout()
    return fig
end

"""Founder search: every chain's trajectory, and where the winner landed."""
function plot_founder(trajectories, final_fitness, founder_index, founder_fitness, ζ)
    n_steps, n_chains = size(trajectories, 1) - 1, size(trajectories, 2)
    fig, axs = subplots(1, 2, figsize=(10, 3.6))

    ax = axs[1]
    steps = 0:n_steps
    for k in 1:n_chains
        @views ax.plot(steps, trajectories[:, k], color="0.5", lw=0.3, alpha=0.35)
    end
    @views ax.plot(steps, trajectories[:, founder_index], color="tab:red", lw=1.2,
                   label="founder")
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel("system fitness")
    ax.set_title("$n_chains chains, \$\\zeta\$ = $ζ", fontsize=10)
    ax.legend(fontsize=7, frameon=false, loc="lower right")

    ax = axs[2]
    ax.hist(final_fitness, bins=30, color="0.4", edgecolor="white", linewidth=0.4)
    ax.axvline(founder_fitness, color="tab:red", lw=1.5, ls="--", label="founder")
    ax.set_xlabel("final fitness")
    ax.set_ylabel("count")
    ax.set_title(@sprintf("%d chains, best %.4f", n_chains, founder_fitness), fontsize=10)

    fig.tight_layout()
    return fig
end

"""The drifted pool: how far the orthologs diverged, and what it cost in fitness."""
function plot_ortholog_pool(similarities, trajectories, founder_fitness, ζ, total_sites)
    n_steps, n_orthologs = size(trajectories, 1) - 1, size(trajectories, 2)
    fig, axs = subplots(1, 2, figsize=(10.5, 4.0))

    ax = axs[1]
    # Full 0-1 colour range, diagonal included: self-similarity is 1, so the diagonal
    # reads as the top of the scale and everything else is judged against it.
    im = ax.imshow(similarities, cmap="viridis", origin="lower", interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_xlabel("ortholog")
    ax.set_ylabel("ortholog")
    ax.set_title("pairwise sequence similarity (over $total_sites sites)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[2]
    steps = 0:n_steps
    for k in 1:n_orthologs
        @views ax.plot(steps, trajectories[:, k], color="0.5", lw=0.4, alpha=0.5)
    end
    ax.axhline(founder_fitness, color="tab:red", lw=1.5, ls="--", label="founder")
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel("system fitness")
    ax.set_ylim(bottom=0)
    ax.set_title("$n_orthologs chains drifting at \$\\zeta\$ = $ζ", fontsize=10)
    ax.legend(fontsize=7, frameon=false, loc="lower left")

    fig.tight_layout()
    return fig
end

"""Per-site conservation across the ortholog pool, in bits."""
function plot_site_conservation(conservation, q, protein_len)
    total_sites = length(conservation)
    fig, ax = subplots(figsize=(9, 2.6))
    ax.bar(1:total_sites, conservation, color="0.4", width=1.0)
    for b in protein_len:protein_len:(total_sites - 1)
        ax.axvline(b + 0.5, color="tab:red", lw=1.2)
    end
    ax.set_xlabel("site (concatenated system; red = protein boundary)")
    ax.set_ylabel("conservation (bits)\n\$\\log_2 q - H\$")
    ax.set_xlim(0.5, total_sites + 0.5)
    ax.set_ylim(0, log2(q))
    ax.set_title("per-site conservation in the ortholog pool", fontsize=10)
    fig.tight_layout()
    return fig
end

"""
Deep mutational scans of both parents on one shared colour scale.

Effects are strongly concentrated under `Binding()`, so a symmetric linear scale set
by the extreme would render everything but the binding site flat grey. The limit is a
high percentile of |ΔF| pooled over both parents, shared by the two panels so they are
directly comparable; beyond it the colour saturates.
"""
function plot_parent_dms(dms1, dms2, p1, p2, parent_fitness, parent_pair,
                         protein_len, q; lim=nothing)
    total_sites = size(dms1, 2)
    lim = something(lim, percentile(abs.(vcat(vec(dms1), vec(dms2))), 99))
    cmap = PyPlot.matplotlib.colors.LinearSegmentedColormap.from_list(
        "dms_green_grey_orange", ["#1b7837", "#e8e8e8", "#e08214"])

    fig, axs = subplots(2, 1, figsize=(13, 6.4), sharex=true)
    for (ax, D, parent, name, f0) in zip(axs, (dms1, dms2), (p1, p2),
                                         ("parent 1 (ortholog $(parent_pair[1]))",
                                          "parent 2 (ortholog $(parent_pair[2]))"),
                                         parent_fitness)
        im = ax.imshow(D, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                       origin="lower", interpolation="nearest",
                       extent=(0.5, total_sites + 0.5, 0.5, q + 0.5))
        # The wild-type residue scores 0 like any neutral substitution, so mark it --
        # otherwise the parent's own sequence is invisible in its own scan.
        ax.plot(1:total_sites, [parent[i] for i in 1:total_sites], ".",
                color="0.25", markersize=2.0)
        # The module boundary, for reference only -- the scan knows nothing about it.
        ax.axvline(protein_len + 0.5, color="k", lw=1.4, ls="--")
        ax.set_ylabel("amino acid")
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_xlim(0.5, total_sites + 0.5)
        ax.set_title(@sprintf("%s, fitness %.4f", name, f0), fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.008,
                     label="\$\\Delta\$fitness", extend="both")
    end
    axs[2].set_xlabel("site in the concatenated system")
    fig.suptitle("deep mutational scan of both parents (dashed = module boundary)",
                 fontsize=11)
    fig.tight_layout()
    return fig
end

"""Library fitness, and how it falls with recombination depth."""
function plot_library_fitness(fitness, crossovers; assay_name="AndGate")
    fig, axs = subplots(1, 2, figsize=(10.5, 3.8))

    ax = axs[1]
    ax.hist(fitness, bins=100, color="0.4", edgecolor="none")
    ax.set_xlabel("chimera fitness ($assay_name)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_title(@sprintf("%.0e chimeras, mean %.4f", length(fitness), mean(fitness)),
                 fontsize=10)

    ax = axs[2]
    xs = crossover_bins(crossovers)
    means = [mean(fitness[crossovers .== k]) for k in xs]
    los = [percentile(fitness[crossovers .== k], 10) for k in xs]
    his = [percentile(fitness[crossovers .== k], 90) for k in xs]
    ax.fill_between(xs, los, his, color="tab:blue", alpha=0.2, label="10-90 pct")
    ax.plot(xs, means, "o-", color="0.3", label="mean fitness")
    ax.set_xlabel("number of crossovers")
    ax.set_ylabel("chimera fitness")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=false)
    ax.set_title("fitness vs. crossover count", fontsize=10)

    fig.tight_layout()
    return fig
end

# ================================================================ step 5 ====

"""
The raw ancestry matrix.

Everything the covariance analysis does is a summary of this picture, so it is worth
looking at directly: each row is one chimera, each column a site, and the colour is
which parent that site was inherited from. The contiguous stretches of one colour ARE
the telegraph process — the structure the null carries is visible here before any
statistics are run. Drawn from the ANALYSED rows only, so the picture matches the
matrices; a panel of intact parents would just be flat stripes no matrix ever sees.
"""
function plot_data_matrix(ancestry, fitness, threshold, shown, labels; note="")
    n_show = length(shown)
    cmap = PyPlot.matplotlib.colors.ListedColormap(["#2166ac", "#b2182b"])

    fig, axs = subplots(1, 2, figsize=(11, 5.2),
                        gridspec_kw=Dict("width_ratios" => [1, 0.17]))
    ax = axs[1]
    ax.imshow(permutedims(ancestry[:, shown]), cmap=cmap, origin="upper",
              interpolation="nearest", aspect="auto", vmin=-0.5, vmax=1.5)
    labels isa Integer ? ax.axvline(labels + 0.5, color="k", lw=1.4) :
                         mark_module_axis!(ax, labels)
    ax.set_xlabel("site")
    ax.set_ylabel("chimera")
    ax.set_title("first $n_show recombinant chimeras of the library$note", fontsize=10, loc="left")

    # Two flat colours, so a two-patch legend says more than a colourbar would.
    Patch = PyPlot.matplotlib.patches.Patch
    ax.legend(handles=[Patch(color="#b2182b", label="parent 1"),
                       Patch(color="#2166ac", label="parent 2")],
              ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              frameon=false, fontsize=8, handlelength=1.2)

    ax = axs[2]
    @views ax.barh(1:n_show, fitness[shown], height=1.0,
                   color=[f >= threshold ? "0.35" : "tab:red" for f in fitness[shown]])
    ax.axvline(threshold, color="tab:blue", lw=1.0, ls=":")
    ax.set_ylim(n_show + 0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xlabel("fitness", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("red = broken", fontsize=8, loc="left")

    fig.tight_layout()
    return fig
end

"""
`C` and its pseudo-inverse for the three subsets, labelled in symbols rather than in
words: the panels are the matrices themselves, and a name would only commit the figure
to one reading of them.

Selection can drive `C` near-singular where it fixes co-inheritance, sending a handful
of `C⁺` entries orders above the rest; on a full-range scale those would flatten the
tridiagonal band the panel is about. The `C⁺` row is therefore scaled to the NULL's
first off-diagonal — the telegraph band, taken from the null because a selected
subset's own off-diagonal is already contaminated by the blow-up. The matrices
themselves are untouched.
"""
function plot_covariance_matrices(results, labels, total_sites; note="",
                                  names=("full", "functional", "nonfunctional"))
    band_lim = maximum(abs, diag(results["full"].Cplus, 1))
    fig, axs = subplots(2, 3, figsize=(12, 7.4))
    for (c, name) in enumerate(names)
        C = copy(results[name].C); C[diagind(C)] .= 0
        im = heat!(axs[1, c], C, "\$C\$ -- $name" * (name == "full" ? "  (NULL)" : ""),
                   labels)
        fig.colorbar(im, ax=axs[1, c], fraction=0.046)
        Cp = copy(results[name].Cplus); Cp[diagind(Cp)] .= 0
        im = heat!(axs[2, c],
                   Cp, "\$C^{+}\$ -- $name  (colour clipped at \$\\pm\$$(round(Int, band_lim)))",
                   labels; lim=band_lim)
        fig.colorbar(im, ax=axs[2, c], fraction=0.046)
    end
    fig.suptitle("ancestry covariance on all $total_sites sites$note",
                 fontsize=11)
    fig.tight_layout()
    return fig
end

"""
The contrast against the null — the primary result of the covariance analysis.

Top row: the covariance contrast. Bottom row: the same contrast taken on the
CONDITIONAL structure, `C⁺(subset) - C⁺(null)`, which strips out the indirect paths
through the ancestry chain. It is the difference of the pseudo-inverses and not the
pseudo-inverse of the difference: each term is a legitimate precision matrix, whereas a
difference of covariances is indefinite, is no distribution's covariance, and inverting
it would weight the directions where the two agree most — the sampling-noise floor.
"""
function plot_contrast(contrast_func, contrast_non, contrast_values,
                       cplus_contrast_func, cplus_contrast_non, cplus_contrast_values,
                       labels, total_sites; note="")
    cplus_lim = percentile(
        [abs(M[i, j]) for M in (cplus_contrast_func, cplus_contrast_non)
         for i in 1:total_sites for j in 1:total_sites if i != j], 99)

    fig, axs = subplots(2, 3, figsize=(13, 7.8))

    im = heat!(axs[1, 1], contrast_func, "\$C\$(functional) \$-\$ \$C\$(null)", labels)
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046)
    im = heat!(axs[1, 2], contrast_non, "\$C\$(non-functional) \$-\$ \$C\$(null)", labels)
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)
    ax = axs[1, 3]
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(1:total_sites, contrast_values, "o-", ms=3, color="0.3")
    ax.set_xlabel("mode (ranked by \$|\\lambda|\$)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("spectrum of the \$C\$ contrast", fontsize=9)

    clip = "  (colour clipped at \$\\pm\$$(round(cplus_lim, sigdigits=2)))"
    im = heat!(axs[2, 1], cplus_contrast_func,
               "\$C^{+}\$(functional) \$-\$ \$C^{+}\$(null)" * clip, labels; lim=cplus_lim)
    fig.colorbar(im, ax=axs[2, 1], fraction=0.046, extend="both")
    im = heat!(axs[2, 2], cplus_contrast_non,
               "\$C^{+}\$(non-functional) \$-\$ \$C^{+}\$(null)" * clip, labels; lim=cplus_lim)
    fig.colorbar(im, ax=axs[2, 2], fraction=0.046, extend="both")
    ax = axs[2, 3]
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(1:total_sites, cplus_contrast_values, "o-", ms=3, color="0.3")
    ax.set_xlabel("mode (ranked by \$|\\lambda|\$)")
    ax.set_ylabel("eigenvalue")
    ax.set_yscale("symlog")
    ax.set_title("spectrum of the \$C^{+}\$ contrast", fontsize=9)

    fig.suptitle("contrast against the unselected null: covariance (top) and " *
                 "conditional (bottom)$note", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig
end

"""
Leading modes of a covariance matrix drawn on the protein structures.

Read these against the null, not on their own: a Bernoulli crossover process alone
gives `Cov(i,j) ~ (1-2p)^|i-j|`, so the null ALREADY has smooth banded modes that
respect the protein boundary, and a selected subset inherits them. Block structure here
is not evidence of modularity.
"""
function plot_modes_on_structure(vectors, values, boundary, protein_len, n_proteins_,
                                 W, L; n_modes=10, title="")
    varfrac = values ./ sum(values)
    n_modes = min(n_modes, length(values))
    fig, axs = subplots(n_modes, n_proteins_,
                        figsize=(1.7 * n_proteins_, 1.5 * n_modes))
    for k in 1:n_modes
        v = vectors[:, k]
        scale = 6 / maximum(abs, v)          # comparable dot sizes across modes
        for p in 1:n_proteins_
            plot_loading!(axs[k, p], scale .* loading_on_protein(v, p, protein_len, W, L))
        end
        # One label per ROW, not per panel: eigenvalue, variance share and protein-1
        # weight are properties of the mode, so repeating them over both proteins only
        # ate the width. Placed at x = 1.05 in axes coordinates, i.e. over the gap.
        axs[k, 1].set_title(
            @sprintf("PC%d   \$\\lambda\$=%.2f (%.1f%% var)   P1 weight %.2f",
                     k, values[k], 100 * varfrac[k], protein1_weight(v, boundary)),
            fontsize=6.5, x=1.05)
    end
    fig.suptitle(title, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig
end

# ================================================================ step 6 ====

"""Duplication, and the split it forces: over unique sequences, not rows."""
function plot_splits(crossovers, group_crossovers, yu, threshold,
                     train_idx, val_idx, test_idx)
    n_rows, n_groups = length(crossovers), length(group_crossovers)
    fig, axs = subplots(1, 2, figsize=(9.5, 3.8))

    ax = axs[1]
    ks = crossover_bins(crossovers; min_count=1, max_bins=20)
    ax.bar(ks .- 0.2, [count(==(k), crossovers) for k in ks], width=0.4,
           color="0.65", label="rows")
    ax.bar(ks .+ 0.2, [count(==(k), group_crossovers) for k in ks], width=0.4,
           color="tab:blue", label="unique sequences")
    ax.set_yscale("log")
    ax.set_xlabel("crossovers")
    ax.set_ylabel("count")
    ax.set_title(@sprintf("only %.0f%% of rows are unique", 100 * n_groups / n_rows),
                 fontsize=10)
    ax.legend(fontsize=7, frameon=false)

    ax = axs[2]
    for (name, idx, c) in (("train", train_idx, "0.35"), ("val", val_idx, "tab:orange"),
                           ("test", test_idx, "tab:blue"))
        ax.hist(yu[idx], bins=80, histtype="step", density=true, color=c, label=name)
    end
    ax.axvline(threshold, color="tab:red", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("fitness")
    ax.set_ylabel("density")
    ax.set_title("the three splits -- same distribution", fontsize=10)
    ax.legend(fontsize=7, frameon=false)

    fig.tight_layout()
    return fig
end

# ================================================================ step 7 ====

# SpinModel's matplotlib style renders text through LaTeX, where an underscore is a
# math-mode character and a bare one aborts the whole figure. Model names are
# dictionary keys and carry underscores, so nothing goes into a figure without passing
# through here.
pretty_name(nm) = replace(nm, "_" => " ")

# The L1 fits are drawn in the same hue as their L2 counterparts, lighter, so a penalty
# pair reads as a pair rather than as two unrelated models.
const MODEL_COLORS = Dict("additive" => "0.45", "additive_lasso" => "0.72",
                          "pairwise" => "tab:blue", "pairwise_lasso" => "tab:cyan",
                          "mlp" => "tab:orange")

"""Accuracy of every model on held-out chimeras, the training curve, and the best
model's predictions against the truth."""
function plot_model_comparison(model_names, primary, histories,
                               y_test, ŷ_best, best_name, crossovers_test)
    fig, axs = subplots(1, 3, figsize=(15.5, 4.1))

    ax = axs[1]
    xs = 1:length(model_names)
    vals = [primary[nm].r2 for nm in model_names]
    ax.bar(xs, vals, width=0.6, color="tab:blue")
    for (xi, v) in zip(xs, vals)
        ax.text(xi, max(v, 0) + 0.015, @sprintf("%.3f", v), ha="center", fontsize=6)
    end
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty_name.(model_names), fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("\$R^2\$ (fitness)")
    ax.set_ylim(0, 1)
    ax.set_title("accuracy on chimeras never seen in training", fontsize=10)

    ax = axs[2]
    for (nm, h) in histories
        ax.plot(1:length(h), h, "-o", ms=2.5, color=get(MODEL_COLORS, nm, "0.3"), label=nm)
    end
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (standardized)")
    ax.set_yscale("log")
    ax.set_title("training", fontsize=10)
    ax.legend(fontsize=7, frameon=false)

    ax = axs[3]
    sc = ax.scatter(y_test, ŷ_best, s=1.5, alpha=0.15, c=crossovers_test,
                    cmap="viridis", rasterized=true)
    lims = [minimum(y_test), maximum(y_test)]
    ax.plot(lims, lims, "-", color="tab:red", lw=1.0)
    ax.set_xlabel("true fitness"); ax.set_ylabel("predicted")
    ax.set_title(@sprintf("%s, held-out chimeras (\$R^2\$ = %.4f)",
                          pretty_name(best_name), primary[best_name].r2), fontsize=10)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="crossovers")

    fig.tight_layout()
    return fig
end

"""
The pairwise model's fitted parameters under both penalties: the couplings as matrices,
the fields beside them, and the per-site coupling strength.

The two fits come from the same Gram, the same features and the same targets; the only
difference between them is L2 against L1. One shared colour scale, so the comparison is
of values and not of two auto-scaled pictures — which matters because the ridge matrix is
dense by construction while the L1 one is exactly zero almost everywhere, and an
independent scale would hide precisely that.

Fields and couplings are the two halves of the same model and are on the same scale, so
they are drawn together: a site can matter because its own spin is favoured (a field) or
because of whom it is coupled to, and the module question is about the second. In spin
coding both are read the Ising way — a positive coupling favours the two sites agreeing,
a positive field favours that site sitting at +1 in this coding.
"""
function plot_interactions(J, J_l1, h, h_l1, labels; ridge_λ=nothing, lasso_ratio=nothing,
                           note="")
    p = size(J, 1)
    n_pairs = p * (p - 1) ÷ 2
    nz_l1 = count(!iszero, J_l1) ÷ 2
    fig, axs = subplots(2, 2, figsize=(12.5, 9.0))
    lim = quantile(abs.(vcat(vec(J), vec(J_l1))), 0.999)
    titles = (ridge_λ === nothing ? "ridge, all $n_pairs pairs nonzero" :
              @sprintf("ridge (\$\\lambda\$ = %g), all %d pairs nonzero", ridge_λ, n_pairs),
              lasso_ratio === nothing ? "lasso, $nz_l1 of $n_pairs nonzero" :
              @sprintf("lasso (\$\\lambda/\\lambda_{max}\$ = %.3g), %d of %d nonzero",
                       lasso_ratio, nz_l1, n_pairs))
    for (c, (M, name)) in enumerate(zip((J, J_l1), titles))
        local ax = axs[1, c]
        im = ax.imshow(M, cmap="RdBu_r", origin="lower", interpolation="nearest",
                       vmin=-lim, vmax=lim)
        mark_modules!(ax, labels)
        ax.set_title("couplings \$J_{ij}\$ -- " * name, fontsize=9)
        ax.set_xlabel("site"); ax.set_ylabel("site")
        fig.colorbar(im, ax=ax, fraction=0.046, extend="both")
    end

    # Fields, signed, both penalties on one axis.
    ax = axs[2, 1]
    ax.bar(1:p, h, color="tab:blue", width=1.0, label="ridge")
    ax.bar(1:p, h_l1, color="tab:red", width=1.0, alpha=0.6, label="lasso")
    ax.axhline(0, color="0.7", lw=0.8)
    mark_module_axis!(ax, labels)
    ax.set_xlabel("site")
    ax.set_ylabel("field \$h_i\$")
    ax.set_title("fitted fields: the site's own term", fontsize=9)
    ax.legend(fontsize=7, frameon=false)

    ax = axs[2, 2]
    ax.bar(1:p, vec(sum(abs, J, dims=2)), color="tab:blue", width=1.0, label="ridge")
    ax.bar(1:p, vec(sum(abs, J_l1, dims=2)), color="tab:red", width=1.0, alpha=0.6,
           label="lasso")
    mark_module_axis!(ax, labels)
    ax.set_xlabel("site")
    # Pipes must be inside math mode; usetex draws a bare | as a dash.
    ax.set_ylabel("\$\\sum_j |J_{ij}|\$")
    ax.set_title("total coupling strength per site", fontsize=9)
    ax.legend(fontsize=7, frameon=false)

    fig.suptitle("the pairwise model's parameters: same Gram, same features, penalty is " *
                 "the only difference$note", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
end

# ================================================================ step 9 ====

"""
One column per matrix: the dendrogram, the matrix in its leaf order, and a strip saying
which module each reordered column came from.

The strip is what makes the picture readable at all — reordering destroys the site
axis, so the boundary can no longer be a line across the matrix and has to travel with
the columns.
"""
function plot_clustered_interactions(results, true_labels, k_show, linkage; lim=nothing)
    p = length(true_labels)
    lim = something(lim, quantile(vcat([abs.(vec(r.J)) for r in results]...), 0.999))
    fig, axs = subplots(3, length(results), figsize=(6.5 * length(results), 8.6),
                        gridspec_kw=Dict("height_ratios" => [0.34, 1.0, 0.08]))

    for (c, r) in enumerate(results)
        pos = zeros(2p - 1)                       # x of every node in the leaf order
        for (x, leaf) in enumerate(r.order)
            pos[leaf] = x
        end
        height_of = zeros(2p - 1)                 # leaves sit at zero
        for m in 1:(p - 1)
            pos[p + m] = (pos[r.merges[m, 1]] + pos[r.merges[m, 2]]) / 2
            height_of[p + m] = r.heights[m]
        end

        # A branch is coloured when every leaf below it belongs to one cluster of the
        # displayed cut, and grey above that, which makes the cut visible without
        # drawing a line at an arbitrary height.
        local ax = axs[1, c]
        for m in 1:(p - 1)
            a, b_ = r.merges[m, 1], r.merges[m, 2]
            h = r.heights[m]
            cl = unique(r.labels_show[r.leaves[p + m]])
            col = length(cl) == 1 ? CLUSTER_COLORS[mod1(cl[1], length(CLUSTER_COLORS))] : "0.55"
            ax.plot([pos[a], pos[a], pos[b_], pos[b_]],
                    [height_of[a], h, h, height_of[b_]], lw=0.8, color=col)
        end
        ax.set_xlim(0.5, p + 0.5)
        ax.set_xticks([])
        ax.set_ylabel("merge height", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(@sprintf("%s -- %s linkage, cophenetic r = %.2f, k=2 adjusted Rand %+.3f",
                              r.name, linkage, r.coph, r.ari), fontsize=9)
        for side in ("top", "right", "bottom")
            ax.spines[side].set_visible(false)
        end

        ax = axs[2, c]
        im = ax.imshow(r.J[r.order, r.order], cmap="RdBu_r", origin="upper",
                       interpolation="nearest", vmin=-lim, vmax=lim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel("sites, clustered order", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, extend="both")
        for x in 1:(p - 1)
            if r.labels_show[r.order[x]] != r.labels_show[r.order[x + 1]]
                ax.axhline(x + 0.5, color="k", lw=0.7)
                ax.axvline(x + 0.5, color="k", lw=0.7)
            end
        end

        ax = axs[3, c]
        ax.imshow(permutedims(true_labels[r.order]), aspect="auto", interpolation="nearest",
                  cmap=PyPlot.matplotlib.colors.ListedColormap(PROTEIN_COLORS),
                  vmin=0.5, vmax=2.5)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_xlabel(@sprintf("site of origin: orange = protein 1, green = protein 2 (the true modules) -- %d label runs along this order, %.0f expected",
                               r.runs[1], r.runs[2]), fontsize=7)
    end

    fig.suptitle("fitted couplings, clustered: sites reordered by the tree, not by " *
                 "site index\nbranch colour = cluster of the k = $k_show cut; the strip " *
                 "below each matrix carries the true module labels", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
end

"""
    plot_clusters_on_structure(labels, k, protein_len, n_proteins, W, L;
                               true_labels, title) -> Figure

Where the clusters actually sit. `labels` is a per-site cluster assignment already pushed
back to TRUE site indices, so a cluster found in scrambled coordinates lands on the
lattice it belongs to.

One row per cluster, one panel per protein, a filled dot at each member site. Drawn this
way rather than as a single coloured map because the interesting question is not "what
colour is site 47" but "does this cluster live in one protein or both" — a row that fills
one lattice and leaves the other empty is a recovered module, and a row that speckles
both is not. Each row is labelled with its size and how it splits across the true
modules, which is the same statement in numbers.
"""
function plot_clusters_on_structure(labels::AbstractVector{<:Integer}, k::Integer,
                                    protein_len::Integer, n_proteins_::Integer,
                                    W::Integer, L::Integer;
                                    true_labels=nothing, title="")
    fig, axs = subplots(k, n_proteins_, figsize=(1.9 * n_proteins_, 1.7 * k),
                        squeeze=false)
    for c in 1:k
        member = Float64.(labels .== c)
        for pr in 1:n_proteins_
            plot_loading!(axs[c, pr], 6 .* loading_on_protein(member, pr, protein_len, W, L))
        end
        share = true_labels === nothing ? "" :
                @sprintf("   protein 1 share %.2f",
                         count(i -> labels[i] == c && true_labels[i] == 1, eachindex(labels)) /
                         max(count(==(c), labels), 1))
        axs[c, 1].set_title(@sprintf("cluster %d   %d sites%s", c, count(==(c), labels), share),
                            fontsize=6.5, x=1.05)
    end
    fig.suptitle(title, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig
end
