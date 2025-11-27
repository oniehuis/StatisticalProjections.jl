struct DummyProjectionModel <: StatisticalProjections.AbstractCPPLS
    projection::Matrix{Float64}
    X_means::Matrix{Float64}
end

@testset "predict applies centering and component selection" begin
    regression_coefficients = Array{Float64}(undef, 2, 2, 2)
    regression_coefficients[:, :, 1] = [1.0 0.0; 0.0 2.0]
    regression_coefficients[:, :, 2] = [0.5 -0.2; 0.3 0.1]
    X_means = reshape([1.0, 2.0], 1, :)
    Y_means = reshape([0.25, -0.5], 1, :)
    cppls = StatisticalProjections.CPPLSLight(
        regression_coefficients,
        X_means,
        Y_means,
        :regression,
    )

    X = [
        1.0 2.0
        2.0 1.0
        3.0 4.0
    ]
    centered = X .- X_means

    expected = Array{Float64}(undef, size(X, 1), size(Y_means, 2), 2)
    for i = 1:2
        expected[:, :, i] = centered * regression_coefficients[:, :, i] .+ Y_means
    end

    preds_full = StatisticalProjections.predict(cppls, X)
    preds_one = StatisticalProjections.predict(cppls, X, 1)

    @test preds_full ≈ expected
    @test preds_one[:, :, 1] ≈ expected[:, :, 1]
    @test size(preds_full) == (size(X, 1), size(Y_means, 2), 2)
    @test_throws DimensionMismatch StatisticalProjections.predict(cppls, X, 3)
end

@testset "predictonehot converts summed predictions to labels" begin
    regression_coefficients = ones(Float64, 1, 2, 1)
    X_means = reshape([0.0], 1, 1)
    Y_means = reshape([0.1, -0.2], 1, :)
    cppls = StatisticalProjections.CPPLSLight(
        regression_coefficients,
        X_means,
        Y_means,
        :regression,
    )

    predictions = zeros(Float64, 3, 2, 2)
    predictions[:, :, 1] = [
        0.9 0.2
        0.1 0.8
        0.4 0.6
    ]
    predictions[:, :, 2] = [
        0.5 0.4
        0.2 0.7
        0.7 0.3
    ]

    summed = sum(predictions, dims = 3)[:, :, 1]
    adjusted = summed .- (size(predictions, 3) - 1) .* cppls.Y_means
    expected_labels = map(argmax, eachrow(adjusted))

    expected_one_hot = zeros(Int, 3, 2)
    for (row, label) in enumerate(expected_labels)
        expected_one_hot[row, label] = 1
    end

    result = StatisticalProjections.predictonehot(cppls, predictions)
    @test result == expected_one_hot
end

@testset "project centers inputs before applying projection" begin
    X_means = reshape([0.5, -1.0], 1, :)
    projection = [
        1.0 0.0
        0.5 2.0
    ]
    dummy = DummyProjectionModel(projection, X_means)
    X = [
        0.5 -1.0
        1.0 0.0
        2.0 1.0
    ]

    centered = X .- X_means
    expected_scores = centered * projection

    scores = StatisticalProjections.project(dummy, X)
    @test scores ≈ expected_scores
end
