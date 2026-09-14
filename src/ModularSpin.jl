"""
    ModularSpin

Multi-protein spin-glass systems with a ground-truth modular decomposition, and
the machinery to shuffle them into chimeras and ask whether the modules can be
inferred back out.

A system is a vector of *uncoupled* spin-glass proteins (`SpinModel`), so the
protein boundaries are the ground-truth module boundaries by construction. A
system-level `SystemAssay` maps the per-protein binding energies to one fitness,
Metropolis Monte Carlo evolves whole systems under it, and a `CrossoverProcess`
recombines two diverged systems into chimeras with recorded ancestry.

Depends on, and does not modify, `SpinModel` (`../../../protevo/SpinModel`).
"""
module ModularSpin

using SpinModel
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Printf
using Flux
using PyPlot

# Types and functions borrowed from SpinModel and used in signatures below.
using SpinModel: Sequence, Settings, Table, Ligands, Perturbation,
                 computeFitness, computeBindingDMS, rand_table, randSeq,
                 Assay, Stability, Binding, Binding1, Binding2, DoubleBinding,
                 Specificity, Allostery, NegativeAllostery, see_J!

include("types.jl")
include("fitness.jl")
include("evolve.jl")
include("crossover.jl")
include("model.jl")
include("library.jl")
include("dataset.jl")
include("moments.jl")
include("scramble.jl")
include("learning.jl")
include("clustering.jl")
include("plotting.jl")
include("reporting.jl")

export
    # types.jl
    ProteinSystem,
    SystemModel,
    n_proteins,
    n_sites,
    sites_per_protein,
    concatenate,
    split_system,
    hamming,
    rand_system,

    # fitness.jl
    SystemAssay,
    AndGate,
    Compensatory,
    system_fitness,
    module_fitnesses,

    # evolve.jl
    evolve_system,
    evolve_ensemble,

    # crossover.jl
    CrossoverProcess,
    BernoulliCrossover,
    draw_ancestry!,
    crossover_points,
    n_crossovers,
    apply_ancestry!,
    chimera,
    polymorphic_sites,

    # model.jl
    build_system_model,
    ligand_field,

    # library.jl
    dms_scan,
    chimera_library,
    site_conservation,
    pairwise_divergence,
    crossover_bins,
    choose_parent_pair,

    # dataset.jl
    functional_cutoff,
    spin_encode,
    encoding_matches_ancestry,
    group_by_sequence,
    fitness_spread_within_groups,
    stratified_split,
    random_split,
    leak_fraction,

    # scramble.jl
    scramble_permutation,
    scramble_rows,
    unscramble,
    unscramble_rows,
    unscramble_both,
    scrambled_module_labels,

    # moments.jl
    ancestry_moments,
    protein1_weight,
    loading_span,
    ar1_precision_check,

    # learning.jl
    r2, auroc, auprc, evaluate, print_metrics,
    design_additive, pair_index, design_pairwise!, gram_pairwise, predict_pairwise,
    ridge_solve, lasso_lambda_max, lasso_solve, lasso_path, center_gram,
    mlp, SiteAttention, train_net!, predict_net,
    model_hyperparameters, fit_models, MODEL_NAMES,

    # clustering.jl
    coupling_distance, profile_distance, linkage_tree, cutree, adjusted_rand,
    cophenetic_correlation, run_test, cross_boundary_share, cluster_and_score,
    cluster_matrix, modular_control,

    # plotting.jl
    save_figure, pretty_name, mark_boundary!, mark_modules!, mark_module_axis!, heat!,
    plot_loading!, loading_on_protein,
    plot_random_fitness, plot_founder, plot_ortholog_pool, plot_site_conservation,
    plot_parent_dms, plot_library_fitness, plot_data_matrix, plot_covariance_matrices,
    plot_contrast, plot_modes_on_structure, plot_splits, plot_model_comparison,
    plot_interactions, plot_clustered_interactions, plot_clusters_on_structure,

    # reporting.jl
    report_landscape, report_founder, report_orthologs, report_dms, report_library,
    report_subsets, report_ar1_check, report_modes, report_dataset, report_models,
    report_interactions, report_clustering

end # module
