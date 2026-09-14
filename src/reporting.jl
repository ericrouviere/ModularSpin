# The console record: one function per stage, printing the numbers that stage exists to
# produce.
#
# In the package rather than in the experiment script for the same reason the figures
# are: a sweep compares experiments, and two experiments that print different
# diagnostics are not comparable. Everything here only prints — the computation is done
# and the results are passed in.

"""Random-system fitness, the unselected reference distribution."""
function report_landscape(fits; label="AndGate")
    @printf("\n%s fitness of random systems (n = %d)\n", label, length(fits))
    @printf("  mean            %8.4f\n", mean(fits))
    @printf("  std             %8.4f\n", std(fits))
    @printf("  median          %8.4f\n", median(fits))
    @printf("  max             %8.4f\n", maximum(fits))
end

"""The founder search: where the chains ended, and which one won."""
function report_founder(final_fitness, founder_index, founder_fitness, founder_modules,
                        seeds)
    @printf("\nfinal fitness: mean %8.4f  max %8.4f\n",
            mean(final_fitness), maximum(final_fitness))
    @printf("\nfounder: chain %d, seed %d, fitness %.4f\n",
            founder_index, seeds[founder_index], founder_fitness)
    @printf("  per-module fitness: %s\n", string(founder_modules))
end

"""The drifted pool: how far it diverged, and what that cost."""
function report_orthologs(founder_fitness, final_fitness, div, total_sites, conservation, q)
    @printf("\nfitness: founder %.4f -> orthologs %.4f ± %.4f (decay %.1f%%)\n",
            founder_fitness, mean(final_fitness), std(final_fitness),
            100 * (1 - mean(final_fitness) / founder_fitness))
    @printf("\npairwise divergence over %d sites: mean %.1f  min %d  max %d\n",
            total_sites, mean(div.upper), minimum(div.upper), maximum(div.upper))
    @printf("mean pairwise sequence similarity: %.1f%%\n", 100 * div.mean_similarity)
    div.mean_divergence < 0.15 &&
        @warn """orthologs are barely diverged; the chimera library will have little
                 for the ancestry analysis to see. Lower the drift ζ or raise the
                 number of drift steps.""" div.mean_divergence
    @printf("\nconservation (bits, max %.2f): mean %.3f  max %.3f at site %d\n",
            log2(q), mean(conservation), maximum(conservation), argmax(conservation))
end

"""Both parents' mutational scans: how concentrated the fitness effects are."""
function report_dms(dms1, dms2, total_sites)
    for (name, D) in (("parent 1", dms1), ("parent 2", dms2))
        colmax = vec(maximum(abs.(D), dims=1))
        @printf("  %s: min %+.3f  max %+.3f   %d / %d sites with any effect (|dF| > 0.01)\n",
                name, minimum(D), maximum(D), count(>(0.01), colmax), total_sites)
    end
end

"""The library as generated: no cutoff is applied here, only description."""
function report_library(fitness, crossovers; max_rows=15)
    @printf("\nfitness: mean %.4f  median %.4f  min %.4f  max %.4f\n",
            mean(fitness), median(fitness), minimum(fitness), maximum(fitness))
    @printf("crossovers per chimera: mean %.2f  max %d\n",
            mean(crossovers), maximum(crossovers))
    for k in crossover_bins(crossovers; min_count=1, max_bins=max_rows)
        sel = crossovers .== k
        n = count(sel)
        n == 0 && continue
        @printf("  %2d crossovers: n = %8d   mean fitness %7.4f   10th pct %7.4f\n",
                k, n, mean(fitness[sel]), percentile(fitness[sel], 10))
    end
end

"""
What the ancestry analysis is being run on: the screening, the cutoff, and the shares
that set the size of the pedestal the screening removes.
"""
function report_subsets(n_chimeras, total_sites, boundary, recombinant, functional,
                        threshold, quantile)
    @printf("library: %d chimeras x %d sites (all sites used)\n", n_chimeras, total_sites)
    @printf("protein boundary after site %d: %d sites in protein 1, %d in protein 2\n",
            boundary, boundary, total_sites - boundary)
    @printf("dropped %d intact parents (%.2f%% of the library, 0 crossovers); %d chimeras analysed\n",
            count(.!recombinant), 100 * mean(.!recombinant), count(recombinant))
    @printf("functional cutoff: fitness >= %.4f (%.0fth percentile of the library); %.1f%% of all rows, %.1f%% of the analysed ones\n",
            threshold, 100 * quantile, 100 * mean(functional),
            100 * mean(functional[recombinant]))
    @printf("intact parents: %.2f%% of the library, but %.2f%% of the functional rows and %.2f%% of the non-functional ones\n",
            100 * mean(.!recombinant),
            100 * count(.!recombinant .& functional) / count(functional),
            100 * count(.!recombinant .& .!functional) / count(.!functional))
end

"""The AR(1) generator check, which belongs on the unconditioned library alone."""
function report_ar1_check(chk)
    @printf("\ngenerator check on the unconditioned library (exact AR(1) values)\n")
    @printf("  interior diagonal  %8.2f   theory %8.2f\n", chk.interior, chk.interior_theory)
    @printf("  corners            %8.2f   theory %8.2f\n", chk.corners, chk.corners_theory)
    @printf("  first off-diagonal %8.2f   theory %8.2f\n", chk.offdiag, chk.offdiag_theory)
    @printf("  max |C+| at |i-j| >= 2: %.3f  (0 = exactly tridiagonal)\n", chk.max_beyond_band)
end

"""
Where the leading modes live. The null is printed beside the functional subset because
a boundary-respecting functional mode means nothing until it is shown to differ from the
one the crossover process gives for free — this table is the only place they meet.
"""
function report_modes(contrast_values, contrast_vectors, functional, null, boundary; n=6)
    println("\nlocalization of leading contrast modes (fraction of weight in protein 1)")
    for k in 1:min(n, length(contrast_values))
        @printf("  mode %d: λ = %+7.4f   protein-1 weight = %.3f\n",
                k, contrast_values[k], protein1_weight(contrast_vectors[:, k], boundary))
    end
    println("\nlocalization of leading C modes (fraction of weight in protein 1)")
    for k in 1:min(n, length(functional.values))
        @printf("  PC%d: functional λ = %6.4f (%4.1f%% var), P1 weight = %.3f   |   null λ = %6.4f, P1 weight = %.3f\n",
                k, functional.values[k],
                100 * functional.values[k] / sum(functional.values),
                protein1_weight(functional.vectors[:, k], boundary),
                null.values[k], protein1_weight(null.vectors[:, k], boundary))
    end
end

"""Deduplication and the splits: what the library's duplication does to a dataset."""
function report_dataset(n_rows, n_groups, crossovers, group_crossovers, fitness, fu,
                        threshold, quantile; max_rows=10)
    @printf("  %d unique sequences of %d rows (%.1f%%)\n",
            n_groups, n_rows, 100 * n_groups / n_rows)
    @printf("\nfunctional cutoff: fitness >= %.4f (%.0fth percentile of the library)\n",
            threshold, 100 * quantile)
    @printf("functional fraction: %.1f%% of rows, %.1f%% of unique sequences\n",
            100 * mean(fitness .>= threshold), 100 * mean(fu))
    @printf("\nduplication by crossover count\n")
    @printf("  %3s %10s %10s %8s\n", "xo", "rows", "unique", "mult")
    for k in crossover_bins(crossovers; min_count=1, max_bins=max_rows)
        rows = count(==(k), crossovers)
        rows == 0 && continue
        uniq = count(==(k), group_crossovers)
        @printf("  %3d %10d %10d %8.1f\n", k, rows, uniq, uniq == 0 ? NaN : rows / uniq)
    end
end

"""The model comparison table."""
function report_models(model_names, primary)
    println("="^78)
    @printf("%-15s %12s %10s %10s %10s\n",
            "model", "R2 held-out", "RMSE", "AUROC", "rho")
    for nm in model_names
        @printf("%-15s %12.4f %10.4f %10.4f %10.3f\n",
                nm, primary[nm].r2, primary[nm].rmse, primary[nm].auroc, primary[nm].spearman)
    end
    println("="^78)
end

"""
How much of the fitted coupling crosses between modules, under each penalty.

Taken from the module labels rather than from a corner of `J`, because once the sites
have been relabelled the cross-module pairs are scattered through the matrix instead of
sitting in an off-diagonal block.
"""
function report_interactions(J, J_l1, labels::AbstractVector{<:Integer})
    n_pairs = size(J, 1) * (size(J, 1) - 1) ÷ 2
    nz_l1 = count(!iszero, J_l1) ÷ 2
    cross(M) = sum(abs(M[i, j]) for i in axes(M, 1) for j in (i + 1):size(M, 2)
                   if labels[i] != labels[j]; init=0.0)
    @printf("\ninteractions: ridge %d of %d pairs nonzero, lasso %d (%.2f%%)\n",
            count(!iszero, J) ÷ 2, n_pairs, nz_l1, 100 * nz_l1 / n_pairs)
    @printf("  cross-module share of |J|: ridge %.1f%%, lasso %.1f%%\n",
            100 * cross(J) / (sum(abs, J) / 2), 100 * cross(J_l1) / (sum(abs, J_l1) / 2))
end

"""One clustered interaction matrix: the cut, the order, and the aggregate share."""
function report_clustering(r, true_labels, k_show)
    p = length(true_labels)
    J = r.J
    total = sum(abs, J) / 2
    @printf("\n--- %s ---\n", r.name)
    @printf("  %d of %d pairs nonzero\n", count(!iszero, J) ÷ 2, p * (p - 1) ÷ 2)
    @printf("  cross-boundary share of |J|: %.1f%%, against %.1f%% +- %.1f%% for the same values placed at random (z = %+.1f)\n",
            100 * r.cross_share[1], 100 * r.cross_share[2], 100 * r.cross_share[3],
            r.cross_share[4])
    @printf("  leaf order: %d runs of module label, against %.1f +- %.1f expected (z = %+.2f)\n",
            r.runs[1], r.runs[2], r.runs[3], r.runs[4])
    @printf("  cophenetic correlation %.3f (1 = the distances are exactly a hierarchy)\n",
            r.coph)
    @printf("  under the %s dissimilarity instead: k=2 sizes %s, adjusted Rand %+.4f, cophenetic %.3f\n",
            r.alt_name, string(Tuple(count(==(c), r.alt_labels2) for c in 1:2)),
            r.ari_alt, r.coph_alt)
    @printf("  k = 2 cut: sizes %s, adjusted Rand vs the true modules %+.4f\n",
            string(Tuple(count(==(c), r.labels2) for c in 1:2)), r.ari)
    for c in 1:2
        m = r.labels2 .== c
        @printf("      cluster %d: %3d sites, %5.1f%% from protein 1, holds %5.1f%% of |J|\n",
                c, count(m), 100 * count(m .& (true_labels .== 1)) / count(m),
                100 * (sum(abs, J[m, m]) / 2) / total)
    end
    @printf("  k = %d cut (display):\n", k_show)
    for c in 1:k_show
        m = r.labels_show .== c
        sites = findall(m)
        @printf("      cluster %d: %3d sites, %5.1f%% of |J|, span %3d-%-3d, protein 1 share %.2f\n",
                c, count(m), 100 * (sum(abs, J[m, m]) / 2) / total,
                minimum(sites), maximum(sites), count(true_labels[sites] .== 1) / count(m))
    end
end
