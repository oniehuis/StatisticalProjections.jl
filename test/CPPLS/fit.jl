@testset "fit_cppls builds diagnostic-rich model" begin
    X = Float64[
        1 0 2
        0 1 2
        1 1 1
        2 3 0
        3 2 1
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 1
        1 0
    ]

    model = StatisticalProjections.fit_cppls(X, Y, 2; gamma=0.5)

    @test model isa StatisticalProjections.CPPLS
    @test size(model.regression_coefficients) == (size(X, 2), size(Y, 2), 2)
    @test size(model.fitted_values) == (size(X, 1), size(Y, 2), 2)
    @test size(model.residuals) == size(model.fitted_values)
    @test size(model.X_scores) == (size(X, 1), 2)
    @test size(model.Y_scores) == (size(Y, 1), 2)
    @test model.X_means ≈ StatisticalProjections.mean(X, dims=1)
    @test model.Y_means ≈ StatisticalProjections.mean(Y, dims=1)
    @test model.gammas ≈ fill(0.5, 2)
    @test length(model.canonical_correlations) == 2
    @test length(model.X_variance) == 2
    @test model.X_total_variance > 0
end

@testset "fit_cppls_light matches regression from full model" begin
    X = Float64[
        2 1
        0 3
        4 5
        1 4
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 1
    ]
    gamma_bounds = (0.2, 0.8)

    full = StatisticalProjections.fit_cppls(X, Y, 2; gamma=gamma_bounds)
    light = StatisticalProjections.fit_cppls_light(X, Y, 2; gamma=gamma_bounds)

    @test light isa StatisticalProjections.CPPLSLight
    @test light.regression_coefficients ≈ full.regression_coefficients
    @test light.X_means ≈ full.X_means
    @test light.Y_means ≈ full.Y_means
end

@testset "process_component! normalizes weights and deflates predictors" begin
    X = Float64[
        2 0
        0 1
        1 3
    ]
    Y = Float64[
        1 0
        0 1
        1 1
    ]
    n_components = 1

    X_predictors, Y_responses, Y_combined, observation_weights, X̄_mean, Ȳ_mean,
    X_deflated, X_loading_weights, X_loadings, Y_loadings, small_norm_flags,
    regression_coefficients, _, _ = StatisticalProjections.cppls_prepare_data(
        X, Y, n_components, nothing, nothing, true)

    initial_weights = [3.0, 4.0]
    X_deflated_original = copy(X_deflated)

    X_scoresᵢ, t_norm, _ = StatisticalProjections.process_component!(1, X_deflated,
        copy(initial_weights), Y_responses, X_loading_weights, X_loadings, Y_loadings,
        regression_coefficients, small_norm_flags, 1e-12, 1e-12, 1e-10)

    normalized_weights = initial_weights / StatisticalProjections.norm(initial_weights)
    expected_scores = X_deflated_original * normalized_weights
    expected_norm = StatisticalProjections.dot(expected_scores, expected_scores)
    expected_Y_loadings = (Y_responses' * expected_scores) / expected_norm
    expected_B = X_loading_weights[:, 1:1] *
        StatisticalProjections.pinv(X_loadings[:, 1:1]' * X_loading_weights[:, 1:1]) *
        Y_loadings[:, 1:1]'

    @test X_loading_weights[:, 1] ≈ normalized_weights
    @test X_scoresᵢ ≈ expected_scores
    @test t_norm ≈ expected_norm
    @test Y_loadings[:, 1] ≈ expected_Y_loadings
    @test regression_coefficients[:, :, 1] ≈ expected_B
end

@testset "process_component! guards zero-norm scores" begin
    X = Float64[
        1 0
        0 1
        1 1
    ]
    Y = Float64[
        1 0
        0 1
        1 0
    ]
    X_predictors, Y_responses, _, _, _, _, X_deflated,
    X_loading_weights, X_loadings, Y_loadings, small_norm_flags,
    regression_coefficients, _, _ = StatisticalProjections.cppls_prepare_data(
        X, Y, 1, nothing, nothing, true)

    X_deflated .= 0  # force zero scores regardless of weights
    initial_weights = [1.0, 2.0]
    tol = 1e-8

    _, t_norm, _ = StatisticalProjections.process_component!(1, X_deflated,
        copy(initial_weights), Y_responses, X_loading_weights, X_loadings, Y_loadings,
        regression_coefficients, small_norm_flags, 1e-12, 1e-12, tol)

    @test t_norm == tol
end
