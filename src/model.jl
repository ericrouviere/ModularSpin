# Building the system every experiment runs on. Previously `build_model()` in the
# analysis config: the same construction with the parameters hardcoded as globals.
# They are arguments here, because varying them is the point of an experiment.

"""
    ligand_field(spec, p, n_ligand_sites, W; scale) -> Vector{Float64}

The field protein `p` sees under the condition `spec` (a tuple of one vector per
protein): its vector, scaled. Entries may be any real numbers. Only the structural
checks live here — a physically odd ligand is still a legal one, but one of the
wrong *shape* would fail deep inside scoring, so it is caught at construction.
"""
function ligand_field(spec, p::Integer, n_lig::Integer, W::Integer; scale::Real=1.0)
    field = spec[p]
    # Both conditions act on the same binding site, so the two must match in length;
    # that shared length IS the site.
    length(field) == n_lig ||
        error("protein $p field vector has $(length(field)) entries, expected $n_lig")
    1 <= n_lig <= W ||
        error("protein $p binding site spans $n_lig spins, outside 1:W = 1:$W")
    all(isfinite, field) ||
        error("protein $p field vector must be finite, got $field")
    return scale * float.(collect(field))
end

"""
    build_system_model(; W, L, q, n_proteins, solvent, ligand, ligand_scale, sh, sJ,
                         assay, seed) -> SystemModel

`n_proteins` uncoupled spin-glass proteins sharing one `Settings` and one table `K`.
`seed` determines that table and nothing else.

`K` is the sequence-to-coupling map, i.e. amino-acid chemistry, which is the same
everywhere — so the proteins are given the same one. What makes them different
proteins is their sequence and their ligands, not their chemistry.

The ligands are parameters, not random variables: protein `p` is handed
`solvent[p]` as ligand condition 1 and `ligand[p]` as condition 2, so the proteins
can face the same ligand or different ones. Each vector's entries are unconstrained
reals and their shared length sets that protein's binding site, which occupies spins
`1:length` of the last layer.
"""
function build_system_model(; W::Integer, L::Integer, q::Integer, n_proteins::Integer,
                            solvent, ligand, ligand_scale::Real=1.0,
                            sh::Real=2.0, sJ::Real=2.0, assay=Binding(),
                            seed::Integer)
    length(solvent) == n_proteins ||
        error("solvent spec gives $(length(solvent)) vectors for $n_proteins proteins")
    length(ligand) == n_proteins ||
        error("ligand spec gives $(length(ligand)) vectors for $n_proteins proteins")

    Q = Settings(W, L, q)
    rng = Xoshiro(seed)

    ligs = ntuple(n_proteins) do p
        n_lig = length(ligand[p])
        site = [CartesianIndex(i, L + 1, 1) for i in 1:n_lig]
        # Condition 1 is the unbound reference and condition 2 the bound state,
        # following SpinModel's convention that ligand 1 is the solvent.
        sol = ligand_field(solvent, p, n_lig, W; scale=ligand_scale)
        lig = ligand_field(ligand, p, n_lig, W; scale=ligand_scale)
        # Binding() scores the difference of the two, so equal vectors make every
        # sequence score exactly 0 — a silent way to flatten the landscape.
        sol == lig &&
            @warn "protein $p has the same field in both conditions; every sequence " *
                  "will score 0 under Binding()" protein=p field=sol
        Ligands([Perturbation(site, sol), Perturbation(site, lig)])
    end

    return SystemModel(Q, ligs, n_proteins; assay, shared_table=true, sh, sJ, rng)
end
