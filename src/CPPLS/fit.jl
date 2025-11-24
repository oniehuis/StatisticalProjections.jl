function process_component!(
    i::Integer,
    X_deflated::AbstractMatrix{<:Real},
    X_loading_weightsᵢ::AbstractVector{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    X_loading_weights::AbstractMatrix{<:Real},
    X_loadings::AbstractMatrix{<:Real},
    Y_loadings::AbstractMatrix{<:Real},
    regression_coefficients::Array{<:Real,3},
    small_norm_flags::AbstractMatrix{Bool},
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    tᵢ_squared_norm_tolerance::Real)

    X_loading_weightsᵢ .= (X_loading_weightsᵢ ./ norm(X_loading_weightsᵢ) 
        .* (abs.(X_loading_weightsᵢ) .>= X_loading_weight_tolerance))

    X_scoresᵢ = X_deflated * X_loading_weightsᵢ
    tᵢ_squared_norm = X_scoresᵢ' * X_scoresᵢ

    if isapprox(tᵢ_squared_norm, 0.0)
        tᵢ_squared_norm += tᵢ_squared_norm_tolerance
    end
    X_loadingsᵢ = (X_deflated' * X_scoresᵢ) / tᵢ_squared_norm
    Y_loadingsᵢ = (Y_responses' * X_scoresᵢ) / tᵢ_squared_norm

    X_deflated .-= X_scoresᵢ * X_loadingsᵢ'

    small_norm_flags[i, :] .= vec(sum(abs.(X_deflated), dims=1) .< X_tolerance)
    X_deflated[:, small_norm_flags[i, :]] .= 0

    X_loading_weights[:, i] .= X_loading_weightsᵢ
    X_loadings[:, i] .= X_loadingsᵢ
    Y_loadings[:, i] .= Y_loadingsᵢ
    regression_coefficients[:, :, i] .= (X_loading_weights[:, 1:i] * 
        pinv(X_loadings[:, 1:i]' * X_loading_weights[:, 1:i]) * Y_loadings[:, 1:i]')

    X_scoresᵢ, tᵢ_squared_norm, Y_loadingsᵢ
end

# """
#     fit_cppls(
#         X::AbstractMatrix{<:Real},
#         Y::AbstractMatrix{<:Real},
#         n_components::Integer;
#         gamma_bounds::Union{AbstractVector{<:NTuple{2, <:Real}}, Northing}=nothing,
#         observation_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
#         Y_auxiliary::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
#         center::Bool=true,
#         X_tolerance::Real=1e-12,
#         X_loading_weight_tolerance::Real=eps(Float64), 
#         gamma_optimization_tolerance::Real=1e-4)

# Fit a Canonical Powered Partial Least Squares (CPPLS) model.

# # Arguments
# - `X`: A matrix of predictor variables (observations × features). `NA`s and `Inf`s are not 
#   allowed.
# - `Y`: A matrix of response variables (observations × targets). `NA`s and `Inf`s are not 
#   allowed.

# # Optional Positional Argument
# - `n_components`: The number of components to extract in the CPPLS model. Defaults to 2.

# # Optional Keyword Arguments
# - `gamma_bounds`: A vector of tuples specifying the lower and upper bounds for the power 
#   parameter (`γ`) optimization. Defaults to `nothing`, which disables power parameter 
#   optimization.
# - `observation_weights`: A vector of individual weights for the observations (e.g., 
#   experimental data or samples). Defaults to `nothing`.
# - `Y_auxiliary: A matrix of auxiliary response variables containing additional information 
#   about the observations. Defaults to `nothing`.
# - `center`: Whether to mean-center the `X` and `Y` matrices. Defaults to `true`.
# - `X_tolerance`: Tolerance for small norms in `X`. Columns of `X` with norms below this 
#   threshold are set to zero during deflation. Defaults to `1e-12`.
# - `X_loading_weight_tolerance`: Tolerance for small weights. Elements of the weight vector 
#   below this threshold are set to zero. Defaults to `eps(Float64)`.
# - `gamma_optimization_tolerance`: Tolerance for the optimization process when determining 
#    the power parameter (`γ`). Defaults to `1e-4`.

# # Returns
# A `CPPLS` object containing the following fields:
# - `regression_coefficients`: A 3D array of regression coefficients for 1, ..., 
#   `n_components`.
# - `X_scores`: A matrix of scores (latent variables) for the predictor matrix `X`.
# - `X_loadings`: A matrix of loadings for the predictor matrix `X`.
# - `X_loading_weights`: A matrix of loading weights for the predictor matrix `X`.
# - `Y_scores`: A matrix of scores (latent variables) for the response matrix `Y`.
# - `Y_loadings`: A matrix of loadings for the response matrix `Y`.
# - `projection`: The projection matrix used to convert `X` to scores.
# - `X_means`: A vector of means of the `X` variables (used for centering).
# - `Y_means`: A vector of means of the `Y` variables (used for centering).
# - `fitted_values`: An array of fitted values for the response matrix `Y`.
# - `residuals`: An array of residuals for the response matrix `Y`.
# - `X_variance`: A vector containing the amount of variance in `X` explained by each 
#    component.
# - `X_total_variance`: The total variance in `X`.
# - `gammas`: The power parameter (`γ`) values obtained during power optimization.
# - `canonical_correlations`: Canonical correlation values for each component.
# - `small_norm_indices`: Indices of explanatory variables with norms close to or equal to 
#    zero.
# - `canonical_coefficients`: A matrix containing the canonical coefficients (`a`) from 
#   canonical correlation analysis (`cor(Za, Yb)`).

# # Notes
# - The CPPLS model is an extension of Partial Least Squares (PLS) that incorporates 
#   canonical correlation analysis (CCA) and power parameter optimization to maximize the 
#   correlation between linear combinations of `X` and `Y`.
# - The power parameter (`γ`) controls the balance between variance maximization and 
#   correlation maximization. It is optimized within the specified bounds (`gamma_bounds`).
# - If `Y_auxiliary` is provided, it is concatenated with `Y` to form a combined response 
#   matrix (`Y_combined`), which is used during the fitting process.

# # Example
# ```julia
# # Predictor matrix (X) and response matrix (Y)
# X = [
#     0.1322  0.07456  0.07784
#    -4.3688 -2.87124 -3.34636
#     4.6382  3.05536  3.58004
#     1.4822  0.99256  1.14884
#    -1.8838 -1.25124 -1.46036
# ]

# Y = [
#     7.1  6.3  5.4
#     6.2  5.8  4.9
#     8.4  7.2  6.5
#     7.8  6.9  5.9
#     6.5  5.9  4.7
# ]

# # Auxiliary response matrix (optional)
# Y_auxiliary = [
#     7.2  6.4  5.5
#     6.3  5.9  5.0
#     8.5  7.3  6.6
#     7.9  7.0  6.0
#     6.6  6.0  4.8
# ]

# # Observation weights (optional)
# observation_weights = [0.56, 0.07, 0.73, 0.75, 0.03]

# # Power parameter limits
# gamma_bounds = [(0.0, 0.2), (0.8, 1.0)]

# # Fit the CPPLS model with 2 components
# cppls = fit_cppls(X, Y, 2, gamma_bounds=gamma_bounds, 
# observation_weights=observation_weights, Y_auxiliary=Y_auxiliary)

# # Print the regression coefficients
# println(cppls.regression_coefficients)
# """

# TO DO: gamma_bounds should become gamma and can be a scalar or a NTuple{2, T1} with the
# two values being different. Default is gamma = 0.5.
function fit_cppls(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    n_components::Integer=2;
    gamma::Union{<:T1, <:NTuple{2, T1}, <:AbstractVector{<:Union{<:T1, <:NTuple{2, T1}}}}=0.5,
    observation_weights::Union{AbstractVector{T2}, Nothing}=nothing,
    Y_auxiliary::Union{AbstractMatrix{T3}, Nothing}=nothing,
    center::Bool=true,
    X_tolerance::Real=1e-12,
    X_loading_weight_tolerance::Real=eps(Float64),
    gamma_optimization_tolerance::Real=1e-4,
    t_squared_norm_tolerance::Real=1e-10
    ) where {T1<:Real, T2<:Real, T3<:Real}

    (X_predictors, Y_responses, Y_combined, observation_weights, X̄_mean, Ȳ_mean,
    X_deflated, X_loading_weights, X_loadings, Y_loadings, small_norm_flags, 
    regression_coefficients, n_samples_X, n_targets_Y) = cppls_prepare_data(X_predictors, 
        Y_responses, n_components, Y_auxiliary, observation_weights, center)

    X_scores = Matrix{Float64}(undef, n_samples_X, n_components)
    canonical_coefficients = Matrix{Float64}(undef, size(Y_combined, 2), n_components)
    max_canonical_correlations = Vector{Float64}(undef, n_components)
    gamma_values = fill(0.5, n_components)
    X_score_norms = Vector{Float64}(undef, n_components) 
    Y_scores = Matrix{Float64}(undef, n_samples_X, n_components)
    fitted_values = Array{Float64}(undef, n_samples_X, n_targets_Y, n_components)

    for i in 1:n_components
        (X_loading_weightsᵢ, max_canonical_correlations[i], canonical_coefficients[:, i], 
            gamma_values[i]) = (compute_cppls_weights(X_deflated, Y_combined, Y_responses, 
            observation_weights, gamma, gamma_optimization_tolerance))

        X_scoresᵢ, tᵢ_squared_norm, Y_loadingsᵢ = process_component!(i, X_deflated, 
            X_loading_weightsᵢ, Y_responses, X_loading_weights, X_loadings, Y_loadings, 
            regression_coefficients, small_norm_flags, X_tolerance, 
            X_loading_weight_tolerance, t_squared_norm_tolerance)
    
        X_scores[:, i] = X_scoresᵢ
        X_score_norms[i] = tᵢ_squared_norm
        Y_scores[:, i] = Y_responses * Y_loadingsᵢ / (Y_loadingsᵢ' * Y_loadingsᵢ)
    
        if i > 1
            Y_scores[:, i] -= X_scores * (X_scores' * Y_scores[:, i] ./ X_score_norms)
        end
        fitted_values[:, :, i] = X_predictors * regression_coefficients[:, :, i]
    end

    fitted_values .+= reshape(repeat(Ȳ_mean, n_samples_X), n_samples_X, length(Ȳ_mean), 
        1)
    Y_residuals = Y_responses .- fitted_values
    projection = X_loading_weights * pinv(X_loadings' * X_loading_weights)
    X_variance_explained = vec(sum(X_loadings .* X_loadings, dims=1)) .* X_score_norms
    X_total_variance = sum(X_predictors .* X_predictors)

    CPPLS(regression_coefficients, X_scores, X_loadings, X_loading_weights, Y_scores, 
        Y_loadings, projection, X̄_mean, Ȳ_mean, fitted_values, Y_residuals, 
        X_variance_explained, X_total_variance, gamma_values, max_canonical_correlations, 
        small_norm_flags, canonical_coefficients)
end


function fit_cppls_light(
    X_predictors::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    n_components::Integer=2;
    gamma::Union{<:T1, <:NTuple{2, T1}}=0.5,
    observation_weights::Union{AbstractVector{T2}, Nothing}=nothing,
    Y_auxiliary::Union{AbstractMatrix{T3}, Nothing}=nothing,
    center::Bool=true,
    X_tolerance::Real=1e-12,
    X_loading_weight_tolerance::Real=eps(Float64),
    gamma_optimization_tolerance::Real=1e-4,
    t_squared_norm_tolerance::Real=1e-10
    ) where {T1<:Real, T2<:Real, T3<:Real}

    (X_predictors, Y_responses, Y_combined, observation_weights, X̄_mean, Ȳ_mean,
    X_deflated, X_loading_weights, X_loadings, Y_loadings, small_norm_flags, 
    regression_coefficients, _, _) = cppls_prepare_data(X_predictors, Y_responses, 
        n_components, Y_auxiliary, observation_weights, center)

    for i in 1:n_components
        X_loading_weightsᵢ, _, _, _ = compute_cppls_weights(X_deflated, Y_combined, 
            Y_responses, observation_weights, gamma, gamma_optimization_tolerance)

        process_component!(i, X_deflated, X_loading_weightsᵢ, Y_responses,
            X_loading_weights, X_loadings, Y_loadings, regression_coefficients,
            small_norm_flags, X_tolerance, X_loading_weight_tolerance, 
            t_squared_norm_tolerance)

    end

    CPPLSLight(regression_coefficients, X̄_mean, Ȳ_mean)
end
