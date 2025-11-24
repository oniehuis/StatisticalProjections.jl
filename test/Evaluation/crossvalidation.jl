function suppress_info(f::Function)
    logger = Base.CoreLogging.SimpleLogger(IOBuffer(), Base.CoreLogging.Error)
    Base.CoreLogging.with_logger(logger) do
        f()
    end
end

const CROSSVAL_X = Float64[
    1.0   0.0   0.5   1.0
    2.0   0.5   1.0   0.0
    3.0   1.0   0.0   0.5
    4.0   1.5   0.5   1.0
    5.0   2.0   1.0   0.0
    6.0   2.5   0.0   0.5
    7.0   3.0   0.5   1.0
    8.0   3.5   1.0   0.0
    9.0   4.0   0.0   0.5
    10.0  4.5   0.5   1.0
    11.0  5.0   1.0   0.0
    12.0  5.5   0.0   0.5
]

const CROSSVAL_Y = [
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
    1 0
    0 1
]

@testset "random_batch_indices builds stratified folds" begin
    strata = [1, 1, 1, 2, 2, 2]
    folds = StatisticalProjections.random_batch_indices(
        strata, 3, StatisticalProjections.MersenneTwister(1))

    @test length(folds) == 3
    @test sort!(reduce(vcat, folds)) == collect(1:length(strata))
    @test all(length(batch) == 2 for batch in folds)
end

@testset "optimize_num_latent_variables selects component count" begin
    selected = StatisticalProjections.optimize_num_latent_variables(
        CROSSVAL_X, CROSSVAL_Y, 1, 2, 2, 0.5, nothing, nothing, true,
        1e-12, eps(Float64), 1e-10, 1e-4, true,
        StatisticalProjections.MersenneTwister(42), false)
    @test selected == 1
end

@testset "nested_cv returns accuracies and component choices" begin
    accuracies, components = suppress_info() do
        StatisticalProjections.nested_cv(
            CROSSVAL_X, CROSSVAL_Y;
            gamma=0.5,
            num_outer_folds=2,
            num_outer_folds_repeats=2,
            num_inner_folds=2,
            num_inner_folds_repeats=2,
            max_components=1,
            rng=StatisticalProjections.MersenneTwister(123),
            verbose=false)
    end

    @test length(accuracies) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in accuracies)
    @test length(components) == 2
    @test all(comp == 1 for comp in components)
end

@testset "nested_cv_permutation shuffles responses" begin
    perms = suppress_info() do
        StatisticalProjections.nested_cv_permutation(
            CROSSVAL_X, CROSSVAL_Y;
            num_outer_folds=2,
            num_outer_folds_repeats=2,
            num_inner_folds=2,
            num_inner_folds_repeats=2,
            max_components=1,
            num_permutations=2,
            rng=StatisticalProjections.MersenneTwister(321),
            verbose=false)
    end

    @test length(perms) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in perms)
end
