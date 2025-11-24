function suppress_info(f::Function)
    logger = Base.CoreLogging.SimpleLogger(IOBuffer(), Base.CoreLogging.Error)
    Base.CoreLogging.with_logger(logger) do
        f()
    end
end

const CROSSVAL_X = Float64[
    1.1   0.5   1.7   2.3
    2.2   1.1   0.4   3.6
    3.3   1.8   2.5   4.9
    4.4   2.6   1.1   6.1
    5.5   3.3   3.2   7.4
    6.6   4.0   1.8   8.8
    7.7   4.6   2.9   9.9
    8.8   5.3   0.7   11.2
    9.9   6.1   3.6   12.5
    11.0  6.8   2.2   13.7
    12.1  7.5   4.3   15.0
    13.2  8.2   1.5   16.3
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
            center=false,
            rng=StatisticalProjections.MersenneTwister(321),
            verbose=false)
    end

    @test length(perms) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in perms)
end
