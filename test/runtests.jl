using ModularSpin
using SpinModel
using Test
using Random
using Statistics
using LinearAlgebra

# A small model, so the suite runs in seconds rather than minutes.
# Default is the generalist assay (two ligands, three conditions with solvent),
# which is what the pipeline uses; `assay=Binding()` drops to a single ligand.
function test_model(; n=2, W=4, L=5, q=5, seed=42, assay=DoubleBinding(), sim=0.3,
                    shared_table=false)
    Q = Settings(W, L, q)
    rng = Xoshiro(seed)
    site = [CartesianIndex(i, L + 1, 1) for i in 1:W]
    la = normalize(randn(rng, W))
    tmp = normalize(randn(rng, W))
    lb = normalize(tmp - (la'tmp) * la)
    ligs = if assay isa Binding
        Ligands([Perturbation(site, zeros(W)), Perturbation(site, la)])
    else
        Ligands([Perturbation(site, zeros(W)),
                 Perturbation(site, normalize(sim * la + (1 - sim) * lb)),
                 Perturbation(site, normalize(sim * la - (1 - sim) * lb))])
    end
    return SystemModel(Q, ligs, n; assay, shared_table, sh=3.0, sJ=3.0, rng)
end

@testset "ModularSpin" begin

@testset "ProteinSystem" begin
    model = test_model()
    rng = Xoshiro(1)
    sys = rand_system(rng, model)

    @test n_proteins(sys) == 2
    @test sites_per_protein(sys) == 4 * 6
    @test n_sites(sys) == 48
    @test length(concatenate(sys)) == 48
    @test split_system(concatenate(sys), 2) == sys
    @test copy(sys) == sys
    @test copy(sys).seqs[1] !== sys.seqs[1]        # a real copy, not an alias
    @test hamming(sys, sys) == 0
    @test hamming(sys, rand_system(Xoshiro(2), model)) > 0

    # split_system must copy, not alias
    flat = concatenate(sys)
    back = split_system(flat, 2)
    flat[1] = flat[1] % model.Q.q + 1
    @test back == sys

    @test_throws ErrorException split_system(zeros(Int8, 7), 2)
    @test ProteinSystem([Int8[1, 2], Int8[3, 4]]) == ProteinSystem((Int8[1, 2], Int8[3, 4]))
end

@testset "SystemModel" begin
    model = test_model()
    @test n_proteins(model) == 2
    @test sites_per_protein(model) == 24
    @test n_sites(model) == 48
    # Independent tables by default, one shared table on request.
    @test model.Ks[1] != model.Ks[2]
    shared = test_model(shared_table=true)
    @test shared.Ks[1] == shared.Ks[2]
    @test all(K -> K === shared.Ks[1], shared.Ks)
    # Reproducible from the seed.
    @test test_model(seed=7).Ks[1] == test_model(seed=7).Ks[1]
    @test test_model(seed=7).Ks[1] != test_model(seed=8).Ks[1]

    # ligand count must match the assay, or SpinModel fails deep inside splatting
    Q = Settings(4, 5, 5)
    site = [CartesianIndex(i, 6, 1) for i in 1:4]
    two = Ligands([Perturbation(site, zeros(4)), Perturbation(site, ones(4))])
    @test_throws ErrorException SystemModel(Q, two, 2; assay=DoubleBinding())
    @test SystemModel(Q, two, 2; assay=Binding()) isa SystemModel
end

@testset "fitness" begin
    model = test_model()
    sys = rand_system(Xoshiro(3), model)
    ϕ = module_fitnesses(sys, model)

    @test length(ϕ) == 2
    @test ϕ[1] ≈ computeFitness(sys.seqs[1], model.Ks[1], model.Q, model.ligs[1], model.assay)
    # the per-protein assay is honoured, and picking a different one changes the score
    @test model.assay isa DoubleBinding
    @test module_fitnesses(sys, test_model(assay=Binding()))[1] !=
          module_fitnesses(sys, model)[1]
    @test system_fitness(AndGate(), sys, model) ≈ minimum(ϕ)
    @test system_fitness(Compensatory(), sys, model) ≈ sum(ϕ)
    # min <= mean, always
    @test system_fitness(AndGate(), sys, model) <=
          system_fitness(Compensatory(), sys, model) / n_proteins(model) + 1e-12

    # batched form agrees with the scalar form
    systems = [rand_system(Xoshiro(i), model) for i in 1:8]
    @test system_fitness(AndGate(), systems, model) ≈
          [system_fitness(AndGate(), s, model) for s in systems]
end

@testset "crossover" begin
    n = 200
    p1 = Int8.(fill(1, n))
    p2 = Int8.(fill(2, n))
    rng = Xoshiro(11)

    @testset "p = 0" begin
        proc = BernoulliCrossover(0.0)
        for _ in 1:20
            sys, anc = chimera(rng, proc, p1, p2, 2)
            @test n_crossovers(anc) == 0
            @test isempty(crossover_points(anc))
            @test all(anc) || !any(anc)                    # one parent only
            @test concatenate(sys) == (all(anc) ? p1 : p2) # exactly that parent
        end
    end

    @testset "p = 1 alternates strictly" begin
        proc = BernoulliCrossover(1.0)
        anc = BitVector(undef, n)
        draw_ancestry!(anc, rng, proc)
        @test all(i -> anc[i] != anc[i-1], 2:n)
        @test n_crossovers(anc) == n - 1
    end

    @testset "p out of range is rejected" begin
        @test_throws ErrorException BernoulliCrossover(-0.01)
        @test_throws ErrorException BernoulliCrossover(1.5)
    end

    @testset "crossover count is binomial" begin
        # One trial per gap, so the mean is exact -- no thinning correction.
        prob = 0.02
        proc = BernoulliCrossover(prob)
        counts = map(1:20_000) do _
            anc = BitVector(undef, n)
            draw_ancestry!(anc, rng, proc)
            n_crossovers(anc)
        end
        @test mean(counts) ≈ prob * (n - 1) rtol = 0.05
        # Binomial rather than Poisson: variance is (1-p) times the mean.
        @test var(counts) ≈ prob * (1 - prob) * (n - 1) rtol = 0.08
    end

    @testset "p = 0.5 makes sites i.i.d." begin
        # Half the gaps flip, so adjacent sites agree half the time.
        proc = BernoulliCrossover(0.5)
        anc = BitVector(undef, n)
        agree = 0
        reps = 2000
        for _ in 1:reps
            draw_ancestry!(anc, rng, proc)
            agree += count(i -> anc[i] == anc[i-1], 2:n)
        end
        @test agree / (reps * (n - 1)) ≈ 0.5 atol = 0.02
    end

    @testset "correlation decays as (1-2p)^d" begin
        prob = 0.05
        proc = BernoulliCrossover(prob)
        reps = 20_000
        acc = zeros(n)
        anc = BitVector(undef, n)
        for _ in 1:reps
            draw_ancestry!(anc, rng, proc)
            spins = 2.0 .* anc .- 1.0            # +/- 1
            for d in 1:(n - 1)
                acc[d] += sum(spins[i] * spins[i + d] for i in 1:(n - d)) / (n - d)
            end
        end
        for d in (1, 5, 20, 50)
            @test acc[d] / reps ≈ (1 - 2prob)^d atol = 0.02
        end
    end

    @testset "ancestry reproduces the sequence" begin
        model = test_model()
        a = concatenate(rand_system(Xoshiro(21), model))
        b = concatenate(rand_system(Xoshiro(22), model))
        proc = BernoulliCrossover(0.05)
        for _ in 1:50
            sys, anc = chimera(rng, proc, a, b, 2)
            @test concatenate(sys) == ifelse.(anc, a, b)
        end
        # ancestry must come from the process, not be re-derived: at sites where the
        # parents agree, the sequence carries no information about the parent
        shared = findall(a .== b)
        @test !isempty(shared)
        @test length(polymorphic_sites(a, b)) == length(a) - length(shared)
    end

    @testset "crossover_points agrees with n_crossovers" begin
        proc = BernoulliCrossover(0.05)
        for _ in 1:50
            anc = BitVector(undef, n)
            draw_ancestry!(anc, rng, proc)
            @test length(crossover_points(anc)) == n_crossovers(anc)
            @test all(i -> anc[i] != anc[i-1], crossover_points(anc))
        end
    end
end

@testset "evolve" begin
    model = test_model()
    sys0 = rand_system(Xoshiro(5), model)

    @testset "greedy climb is monotone" begin
        # Metropolis accepts a downhill step of size Δ with probability exp(ζΔ), so
        # "monotone" only holds up to steps small enough that ζΔ is still ~0. At
        # ζ = 1e6, steps of order 1e-6 do get through; at ζ = 1e12 nothing above
        # float noise does.
        _, traj = evolve_system(Xoshiro(9), sys0, model, AndGate(), 300, 1e12)
        @test all(>=(-1e-10), diff(traj))
        @test traj[end] > traj[1]
        @test length(traj) == 301

        # ...and at a merely large ζ the walk is monotone in effect, but not exactly.
        _, warm = evolve_system(Xoshiro(9), sys0, model, AndGate(), 300, 1e6)
        @test all(>=(-1e-4), diff(warm))
        @test warm[end] > warm[1]
    end

    @testset "neutral walk is not monotone" begin
        final, traj = evolve_system(Xoshiro(9), sys0, model, AndGate(), 300, 0.0)
        @test !issorted(traj)
        @test final != sys0                         # it moved
    end

    @testset "input is not mutated" begin
        before = copy(sys0)
        evolve_system(Xoshiro(9), sys0, model, AndGate(), 100, 100.0)
        @test sys0 == before
    end

    @testset "trajectory matches recomputed fitness" begin
        final, traj = evolve_system(Xoshiro(13), sys0, model, AndGate(), 200, 200.0)
        @test traj[1] ≈ system_fitness(AndGate(), sys0, model)
        @test traj[end] ≈ system_fitness(AndGate(), final, model)
    end

    @testset "reproducible from seed" begin
        a, ta = evolve_system(Xoshiro(77), sys0, model, AndGate(), 200, 200.0)
        b, tb = evolve_system(Xoshiro(77), sys0, model, AndGate(), 200, 200.0)
        @test a == b
        @test ta == tb
    end

    @testset "ensemble" begin
        seeds = 1:8
        systems, trajs = evolve_ensemble(model, AndGate(), 150, 200.0, seeds)
        @test length(systems) == 8
        @test size(trajs) == (151, 8)
        # each chain reproduces its own seed's single-chain run
        for k in 1:8
            rng = Xoshiro(seeds[k])
            start = rand_system(rng, model)
            s, t = evolve_system(rng, start, model, AndGate(), 150, 200.0)
            @test s == systems[k]
            @test t == trajs[:, k]
        end

        # founder form: every chain starts from the same system
        founder = sys0
        systems2, trajs2 = evolve_ensemble(founder, model, AndGate(), 150, 200.0, seeds)
        @test all(t ≈ system_fitness(AndGate(), founder, model) for t in trajs2[1, :])
        @test founder == sys0                       # founder untouched
    end
end

end # testset ModularSpin
