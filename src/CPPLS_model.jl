abstract type AbstractCPPLS end


struct CPPLS{T1<:Real, T2<:Integer} <: AbstractCPPLS
    regression_coefficients::Array{T1, 3}
    X_scores::Matrix{T1}
    X_loadings::Matrix{T1}
    X_loading_weights::Matrix{T1}
    Y_scores::Matrix{T1}
    Y_loadings::Matrix{T1}
    projection::Matrix{T1}
    X_means::Matrix{T1}
    Y_means::Matrix{T1}
    fitted_values::Array{T1, 3}
    residuals::Array{T1, 3}
    X_variance::Vector{T1}
    X_total_variance::T1
    gammas::Vector{T1}
    canonical_correlations::Vector{T1}
    small_norm_indices::Matrix{T2}
    canonical_coefficients::Matrix{T1}
end


struct CPPLSLight{T1<:Real} <: AbstractCPPLS
    regression_coefficients::Array{T1, 3}
    X_means::Matrix{T1}
    Y_means::Matrix{T1}
end


function center_mean(M::AbstractMatrix{<:Real}, observation_weights::AbstractVector{<:Real})
    M̄ = Matrix((observation_weights' * M) / sum(observation_weights))
    M .- M̄, M̄
end


function center_mean(M::AbstractMatrix{<:Real}, ::Nothing)
    M̄ = mean(M, dims=1)
    M .- M̄, M̄
end



# Function to create a rectangular identity matrix
function Iᵣ(rowcount::Integer, columncount::Integer)
    M = zeros(rowcount, columncount)
    @inbounds for i in 1:min(rowcount, columncount)
        M[i, i] = 1
    end
    M
end


function cca_decomposition(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, 
    ::Nothing)
    
    cca_decomposition(X, Y)
end


function cca_decomposition(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, 
    observation_weights::AbstractVector{<:Real})
    
    cca_decomposition(X .* observation_weights, Y .* observation_weights)
end


function cca_decomposition(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    # Get the number of rows and columns in the predictor matrix X
    n_rows, n_cols = size(X)

    # Perform QR decomposition with column pivoting on both X and Y
    # This step orthogonalizes the columns of X and Y while preserving their rank
    qx = qr(X, ColumnNorm())
    qy = qr(Y, ColumnNorm())

    # Compute the rank of X and Y from the R matrices of the QR decompositions
    dx = rank(qx.R)
    dy = rank(qy.R)
    
    # Ensure that both X and Y have non-zero rank
    @inbounds  if dx == 0
        throw(ErrorException("X has rank 0"))
    end
    @inbounds if dy == 0
        throw(ErrorException("Y has rank 0"))
    end

    # Perform Singular Value Decomposition (SVD) on the product of the orthogonalized 
    # matrices with only singular values. This step computes the canonical correlations 
    # between X and Y
    A = ((qx.Q' * qy.Q) * Iᵣ(n_rows, dy))[1:dx, :]
    left_singular_vecs, singular_vals, _ = svd(A)

    # Extract the maximum canonical correlation (the largest singular value)
    max_canonical_correlation = clamp(first(singular_vals), 0.0, 1.0)

    # Return the decomposition results, including dimensions, QR decomposition results,
    # left singular vectors, and the maximum canonical correlation
    n_rows, n_cols, qx, dx, dy, left_singular_vecs, max_canonical_correlation
end


function cca_corr(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing})
    # Extract the maximum canonical correlation between X and Y
    # by calling the cca_decomposition function and returning its last result.
    last(cca_decomposition(X, Y, observation_weights))
end


function cca_coeffs_and_corr(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, 
    observation_weights::Union{AbstractVector{<:Real}, Nothing})
    # Perform Canonical Correlation Analysis (CCA) decomposition to extract key components
    ((n_rows, n_cols, qx, dx, dy, left_singular_vectors, max_canonical_correlation) 
        = cca_decomposition(X, Y, observation_weights))

    # Compute the canonical coefficients by solving the upper triangular system
    canonical_coefficients = qx.R[1:dx, 1:dx] \ left_singular_vectors
    canonical_coefficients *= sqrt(n_rows - 1)

    # Add rows of zeros if necessary to match the dimensions of the original data
    remaining_rows = n_cols - size(canonical_coefficients, 1)
    if remaining_rows > 0
        canonical_coefficients = vcat(canonical_coefficients, zeros(remaining_rows, 
            min(dx, dy)))
    end

    # Apply the inverse permutation to restore the original order of the coefficients
    canonical_coefficients[invperm(qx.p), :], max_canonical_correlation
end


function cca_coeffs(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, 
    observation_weights::Union{AbstractVector{<:Real}, Nothing})
    # Extract the canonical coefficients between X and Y
    # by calling the cca_coeffs_and_corr function and returning its first result.
    first(cca_coeffs_and_corr(X, Y, observation_weights))
end



centerscale(M::AbstractMatrix{<:Real}, observation_weights::AbstractVector{<:Real}
    ) = (M .- (observation_weights' * M) / sum(observation_weights)) .* observation_weights


centerscale(M::AbstractMatrix{<:Real}, ::Nothing) = M .- mean(M, dims=1)


function correlation(
    X_deflated::AbstractMatrix{<:Real},
    Y_combined::AbstractMatrix{<:Real}, 
    observation_weights::Union{AbstractVector{<:Real}, Nothing})

    correlation(centerscale(X_deflated, observation_weights), 
        centerscale(Y_combined, observation_weights))
end


function correlation(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n = size(X, 1)  # Number of rows in X (and Y)

    # Compute standard deviations for columns of X
    X_standard_deviations = sqrt.(mean(X .^ 2, dims=1))

    # Handle zero standard deviations in X_standard_deviations
    zero_std_mask = vec(X_standard_deviations .== 0.0)
    X_standard_deviations[zero_std_mask] .= 1

    # Compute column norms for Y
    col_norms = sqrt.(mean(Y .^ 2, dims=1))

    # Compute the correlation matrix
    X_Y_correlations = (X' * Y) ./ (n * (X_standard_deviations' * col_norms))

    # Restore zero standard deviations in X_standard_deviations
    X_standard_deviations[zero_std_mask] .= 0

    # Zero out rows in X_Y_correlations corresponding to zero-variance columns in X
    X_Y_correlations[zero_std_mask, :, :] .= 0

    # Return the correlation matrix and the standard deviations of X
    X_Y_correlations, X_standard_deviations
end


function compute_variance_weights(X_standard_deviations::AbstractMatrix{<:Real}
    )::Matrix{Float64}
    # Create a binary mask for the maximum value in X_standard_deviations. This identifies 
    # the positions in X_standard_deviations where the value equals the maximum
    mask = X_standard_deviations .== maximum(X_standard_deviations)

    # Retain only the maximum value(s) in X_standard_deviations by applying the mask
    # The mask ensures that all non-maximum values are set to 0
    # Transpose the transformed X_standard_deviations
    (mask .* X_standard_deviations)'
end


function compute_correlation_weights(X_Y_correlations::AbstractMatrix{<:Real})
    # Create a binary mask for the maximum value in X_Y_correlations
    # This identifies the positions in X_Y_correlations where the value equals the maximum
    mask = X_Y_correlations .== maximum(X_Y_correlations)

    # Compute w as the sum of all maximum values in X_Y_correlations along the rows
    # The mask ensures that only the maximum values contribute to the sum
    sum(mask .* X_Y_correlations, dims=2)
end


function compute_general_weights(
    X_standard_deviations::AbstractMatrix{<:Real}, 
    X_Y_correlations::AbstractMatrix{<:Real},
    gamma::Real, 
    correlation_signs::AbstractMatrix{<:Real})

    # Transform X_standard_deviations using the power parameter gamma
    transformed_X_standard_deviations = X_standard_deviations .^ ((1 - gamma) / gamma)

    # Transform X_Y_correlations using the power parameter gamma
    transformed_X_Y_correlations = X_Y_correlations .^ (gamma / (1 - gamma))

    # Compute the weighted product of transformed_X_Y_correlations and 
    # transformed_X_standard_deviations
    (correlation_signs .* transformed_X_Y_correlations) .* transformed_X_standard_deviations'
end


function evaluate_canonical_correlation(
    gamma::Real,
    X_deflated::AbstractMatrix{<:Real}, 
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing})

    # Determine the appropriate initial weight vector based on the value of gamma
    initial_weights = if gamma == 0
        compute_variance_weights(X_standard_deviations)
    elseif gamma == 1
        compute_correlation_weights(X_Y_correlations)
    else
        compute_general_weights(X_standard_deviations, X_Y_correlations, gamma, 
            correlation_signs)
    end

    # Compute the X_projected as the product of X_deflated and initial_weights.
    # X_projected represents the projection of X_deflated onto the space defined by 
    # initial_weights.
    X_projected = X_deflated * initial_weights


    # Perform canonical correlation analysis (CCA) on X_projected and Y_responses.
    max_canonical_correlation = cca_corr(X_projected, Y_responses, observation_weights)

    # Return the negative squared canonical correlation.
    -max_canonical_correlation^2
end


function compute_best_gamma(
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing}, 
    gamma_bounds::NTuple{2, <:Real},
    gamma_optimization_tolerance::Real)

    result = optimize(gamma -> evaluate_canonical_correlation(gamma, X_deflated, 
        X_standard_deviations, X_Y_correlations, correlation_signs, Y_responses, 
        observation_weights), first(gamma_bounds), last(gamma_bounds), Brent();
        rel_tol=gamma_optimization_tolerance, abs_tol=gamma_optimization_tolerance) 

    # Check if the optimization was successful
    Optim.converged(result) || @warn("gamma optimization failed to converge.")

    # Store the optimized gamma value and the corresponding canonical correlation.
    gamma = result.minimizer
    canonical_correlation = -result.minimum

    # println(gamma)
    # println(canonical_correlation)

    gamma, canonical_correlation
end



function compute_best_gamma(
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing}, 
    gamma_bounds::AbstractVector{<:Union{NTuple{2, <:Real}, Real}},
    gamma_optimization_tolerance::Real)

    n = length(gamma_bounds)
    gamma_values = zeros(Float64, n)
    canonical_correlations = zeros(Float64, n)

    for i in 1:n
        if gamma_bounds[i] isa NTuple{2, <:Real}
            if first(gamma_bounds[i]) ≠ last(gamma_bounds[i])
                gamma_values[i], canonical_correlations[i] = compute_best_gamma(X_deflated, 
                    X_standard_deviations, X_Y_correlations, correlation_signs, Y_responses, 
                    observation_weights, gamma_bounds[i], gamma_optimization_tolerance)
            else
                gamma_values[i] = first(gamma_bounds[i])
                canonical_correlations[i] = evaluate_canonical_correlation(
                    gamma_values[i], X_deflated, X_standard_deviations, X_Y_correlations, 
                    correlation_signs, Y_responses, observation_weights)
            end
        else
            gamma_values[i] = gamma_bounds[i]
            canonical_correlations[i] = evaluate_canonical_correlation(
                gamma_values[i], X_deflated, X_standard_deviations, X_Y_correlations, 
                correlation_signs, Y_responses, observation_weights)
        end
    end

    gamma = gamma_values[argmin(canonical_correlations)]
    canonical_correlation = maximum(-canonical_correlations)

    gamma, canonical_correlation
end


function compute_best_loadings(
    X_deflated::AbstractMatrix{<:Real},
    X_standard_deviations::AbstractMatrix{<:Real},
    X_Y_correlations::AbstractMatrix{<:Real},
    correlation_signs::AbstractMatrix{<:Real},
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing}, 
    gamma_bounds::Union{<:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}},
    gamma_optimization_tolerance::Real,
    n_targets_Y_combined::Integer)

    # Step 1: Preprocess observation_weights
    # Take the square root of the observation_weights to prepare for weighted computations.
    # This ensures that `observation_weights` are applied correctly in subsequent 
    # calculations.
    observation_weights = (isnothing(observation_weights) ? observation_weights : 
        sqrt.(observation_weights))

    gamma, canonical_correlation = compute_best_gamma(X_deflated, X_standard_deviations, 
        X_Y_correlations, correlation_signs, Y_responses, observation_weights,
        gamma_bounds, gamma_optimization_tolerance)

    # Step 5: Compute the optimal loadings based on the selected gamma
    if gamma == 0
        # If gamma = 0, use the standard deviations to compute the loadings.
        optimal_loadings = vec(compute_variance_weights(X_standard_deviations))

        # Return a placeholder matrix for canonical coefficients since no CCA is 
        # performed.
        canonical_coefficients = fill(NaN, (n_targets_Y_combined, 1))
    elseif gamma == 1
        # If gamma = 1, use the `X_Y_correlations` to compute the loadings.
        optimal_loadings = vec(compute_correlation_weights(X_Y_correlations))

        # Return a placeholder matrix for canonical coefficients since no CCA is 
        # performed.
        canonical_coefficients = fill(NaN, (n_targets_Y_combined, 1))
    else
        # For 0 < gamma < 1, compute the optimal loadings using the general formula.
        initial_weights = compute_general_weights(X_standard_deviations, X_Y_correlations, 
            gamma, correlation_signs)

        # Perform canonical correlation analysis (CCA) to compute the canonical 
        # coefficients
        X_projected = X_deflated * initial_weights
        canonical_coefficients = cca_coeffs(X_projected, Y_responses, observation_weights)

        # Compute the final loadings by projecting the canonical coefficients onto the 
        # transformed space.
        optimal_loadings = vec((initial_weights * canonical_coefficients[:, 1])')
    end

    # Step 6: Return the results
    # Return the optimal loadings, the maximum canonical correlation, the first canonical 
    # coefficient vector, and the optimized gamma value.
    # maximum(-canonical_correlations),
    optimal_loadings, canonical_correlation, canonical_coefficients[:, 1], gamma
end


# Function to compute optimal loadings and canonical correlations
function compute_cppls_weights(
    X_deflated::AbstractMatrix{<:Real},
    Y_combined::AbstractMatrix{<:Real}, 
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma::Real,
    gamma_optimization_tolerance::Real)

    if gamma == 0.5

        # Compute the cross-product of X_deflated and Y_combined
        #  supervised dimension reduction 
        initial_weights = X_deflated' * Y_combined

        # Compute canonical correlations between columns in X_deflated * W₀ and Y_responses 
        # with rows weighted according to 'observation_weights'
        ((canonical_coefficients, max_canonical_correlation) = cca_coeffs_and_corr(
            X_deflated * initial_weights, Y_responses, observation_weights))

        # Compute optimal loadings
        optimal_loadings = initial_weights * canonical_coefficients[:, 1]

        # Return the optimal loadings, squared canonical correlation, and first canonical 
        # coefficient vector, and the implicit gamma value of 0.5
        return optimal_loadings, max_canonical_correlation^2, canonical_coefficients[:, 1], 0.5
    else
        return compute_cppls_weights(X_deflated, Y_combined, Y_responses, observation_weights,
            (gamma, gamma), gamma_optimization_tolerance)
    end
end


function compute_cppls_weights(
    X_deflated::AbstractMatrix{<:Real},
    Y_combined::AbstractMatrix{<:Real}, 
    Y_responses::AbstractMatrix{<:Real},
    observation_weights::Union{AbstractVector{<:Real}, Nothing},
    gamma::Union{<:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}},
    gamma_optimization_tolerance::Real)

    # Step 1: Compute correlations and standard deviations
    # C: Correlation matrix between columns of X and Y
    # S: Standard deviations of columns of X
    X_Y_correlations, X_standard_deviations = correlation(X_deflated, Y_combined, 
        observation_weights)
    
    # Step 2: Process the correlation matrix
    # Extract the signs of the correlation matrix
    correlation_signs = sign.(X_Y_correlations)
    
    # Normalize the absolute values of the correlation matrix
    X_Y_correlations = abs.(X_Y_correlations) ./ maximum(X_Y_correlations)

    # Step 3: Normalize the standard deviations
    X_standard_deviations /= maximum(X_standard_deviations)

    # Step 4: Compute the best vector of loadings
    compute_best_loadings(X_deflated, X_standard_deviations, X_Y_correlations, 
        correlation_signs, Y_responses, observation_weights, gamma, 
        gamma_optimization_tolerance, size(Y_combined, 2))
end


function convert_to_float64(M::AbstractMatrix{T}) where T<:Real
    T ≠ Float64 ? convert(Matrix{Float64}, M) : M
end

function cppls_prepare_data(
    X_predictors::AbstractMatrix{<:Real}, 
    Y_responses::AbstractMatrix{<:Real},
    n_components::Integer, 
    Y_auxiliary::Union{AbstractMatrix{<:Real}, Nothing}, 
    observation_weights::Union{AbstractVector{<:Real}, Nothing}, 
    center::Bool)

    X_predictors = convert_to_float64(X_predictors)
    Y_responses = convert_to_float64(Y_responses)

    if Y_auxiliary !== nothing
        Y_auxiliary = convert_to_float64(Y_auxiliary)
    end

    # Step 1: Generate the combined response matrix (Y_combined)
    # Concatenate Y_responses and Y_auxiliary into a single matrix
    Y_combined = isnothing(Y_auxiliary) ? Y_responses : hcat(Y_responses, Y_auxiliary)

    # Step 2: Validate dimensions
    n_samples_X, n_features_X = size(X_predictors)
    n_samples_Y, n_targets_Y = size(Y_responses)
    n_samples_X ≠ n_samples_Y && throw(DimensionMismatch(
        "Number of rows in X_predictors and Y_responses must be equal"))
    if !isnothing(observation_weights) && length(observation_weights) ≠ n_samples_X
        throw(DimensionMismatch("Length of observation_weights must match the number of " * 
        "rows in X_predictors and Y_responses"))
    end

    # Step 3: Center X and Y if required
    if center
        X_predictors, X̄_mean = center_mean(X_predictors, observation_weights)
        Ȳ_mean = mean(Y_responses, dims=1)
    else
        # If not centering, initialize mean vectors as zero
        X̄_mean = zeros(Float64, (1, n_features_X))
        Ȳ_mean = zeros(Float64, (1, n_targets_Y))
    end

     # Step 4: Initialize variables for CPPLS
     X_deflated = copy(X_predictors)
     X_loading_weights = Matrix{Float64}(undef, n_features_X, n_components)
     X_loadings = Matrix{Float64}(undef, n_features_X, n_components)
     Y_loadings = Matrix{Float64}(undef, n_targets_Y, n_components)
     small_norm_flags = Matrix{Bool}(undef, (n_components, n_features_X))
     regression_coefficients = Array{Float64}(undef, (n_features_X, n_targets_Y, 
         n_components)) 


    (X_predictors, Y_responses, Y_combined, observation_weights, X̄_mean, Ȳ_mean,
    X_deflated, X_loading_weights, X_loadings, Y_loadings, small_norm_flags, 
    regression_coefficients, n_samples_X, n_targets_Y)
end


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
    X_loading_weight_tolerance::Real
)
    # Normalize and apply tolerance
    X_loading_weightsᵢ .= (X_loading_weightsᵢ ./ norm(X_loading_weightsᵢ) 
        .* (abs.(X_loading_weightsᵢ) .>= X_loading_weight_tolerance))

    # Compute scores and loadings
    println("X_deflated: ")
    println(any(isnan, X_deflated))

    println("X_loading_weightsᵢ: ")
    println(any(isnan, X_loading_weightsᵢ))

    X_scoresᵢ = X_deflated * X_loading_weightsᵢ

    println("X_scoresᵢ: ")
    println(any(isnan, X_scoresᵢ))

    tᵢ_squared_norm = X_scoresᵢ' * X_scoresᵢ

    if isapprox(tᵢ_squared_norm, 0.0)
        tᵢ_squared_norm += 1e-10
    end

    println(tᵢ_squared_norm)

    println("tᵢ_squared_norm: ")
    println(any(isnan, tᵢ_squared_norm))

    X_loadingsᵢ = (X_deflated' * X_scoresᵢ) / tᵢ_squared_norm
    
    println("X_loadingsᵢ: ")
    println(any(isnan, X_loadingsᵢ))

    Y_loadingsᵢ = (Y_responses' * X_scoresᵢ) / tᵢ_squared_norm

    # Deflate
    X_deflated .-= X_scoresᵢ * X_loadingsᵢ'

    # Zero out small norm columns
    small_norm_flags[i, :] .= vec(sum(abs.(X_deflated), dims=1) .< X_tolerance)
    X_deflated[:, small_norm_flags[i, :]] .= 0

    # Store results
    X_loading_weights[:, i] .= X_loading_weightsᵢ
    X_loadings[:, i] .= X_loadingsᵢ
    Y_loadings[:, i] .= Y_loadingsᵢ

    M = X_loadings[:, 1:i]' * X_loading_weights[:, 1:i]

    if rank(M) < min(size(M)...)
        @warn "Rank-deficient matrix at component $i — using pinv()"
        M_inv = pinv(M)  # handles singular/near-singular safely
    else
        M_inv = M \ I  # efficient solve when full rank
    end

    regression_coefficients[:, :, i] .= X_loading_weights[:, 1:i] * M_inv * Y_loadings[:, 1:i]'

    # regression_coefficients[:, :, i] .= (
    #     X_loading_weights[:, 1:i] *
    #     (inv(X_loadings[:, 1:i]' * X_loading_weights[:, 1:i]) * Y_loadings[:, 1:i]')
    # )

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
    gamma_optimization_tolerance::Real=1e-4
    ) where {T1<:Real, T2<:Real, T3<:Real}

    (X_predictors, Y_responses, Y_combined, observation_weights, X̄_mean, Ȳ_mean,
    X_deflated, X_loading_weights, X_loadings, Y_loadings, small_norm_flags, 
    regression_coefficients, n_samples_X, n_targets_Y) = cppls_prepare_data(X_predictors, 
        Y_responses, n_components, Y_auxiliary, observation_weights, center)

    # Initialize additional variables
    X_scores = Matrix{Float64}(undef, n_samples_X, n_components)
    canonical_coefficients = Matrix{Float64}(undef, size(Y_combined, 2), n_components)
    max_canonical_correlations = Vector{Float64}(undef, n_components)
    gamma_values = fill(0.5, n_components)
    X_score_norms = Vector{Float64}(undef, n_components) 
    Y_scores = Matrix{Float64}(undef, n_samples_X, n_components)
    fitted_values = Array{Float64}(undef, n_samples_X, n_targets_Y, n_components)

    # Step 5: Iteratively compute components
    for i in 1:n_components
        # Compute optimal loading weights and canonical correlations, canonical 
        # coefficients, and if the power algorithm is applied, also optimal gamma values
        (X_loading_weightsᵢ, max_canonical_correlations[i], canonical_coefficients[:, i], 
            gamma_values[i]) = (compute_cppls_weights(X_deflated, Y_combined, Y_responses, 
            observation_weights, gamma, gamma_optimization_tolerance))

        X_scoresᵢ, tᵢ_squared_norm, Y_loadingsᵢ = process_component!(i, X_deflated, 
            X_loading_weightsᵢ, Y_responses, X_loading_weights, X_loadings, Y_loadings, 
            regression_coefficients, small_norm_flags, X_tolerance, 
            X_loading_weight_tolerance)
    
        # Store additional results
        X_scores[:, i] = X_scoresᵢ
        X_score_norms[i] = tᵢ_squared_norm
        Y_scores[:, i] = Y_responses * Y_loadingsᵢ / (Y_loadingsᵢ' * Y_loadingsᵢ)

        if i > 1
            # Orthogonalize Y_scores with respect to previous X_scores
            Y_scores[:, i] -= X_scores * (X_scores' * Y_scores[:, i] ./ X_score_norms)
        end
        fitted_values[:, :, i] = X_predictors * regression_coefficients[:, :, i]
    end

    # Compute residuals and variance explained
    fitted_values .+= reshape(repeat(Ȳ_mean, n_samples_X), n_samples_X, length(Ȳ_mean), 
        1)
    Y_residuals = Y_responses .- fitted_values

    M = X_loadings' * X_loading_weights
    projection = if rank(M) < min(size(M)...)
        @warn "Rank-deficient projection matrix — using pinv"
        X_loading_weights * pinv(M)
    else
        X_loading_weights * (M \ I)
    end

    # projection = X_loading_weights * inv(X_loadings' * X_loading_weights)
    X_variance_explained = vec(sum(X_loadings .* X_loadings, dims=1)) .* X_score_norms
    X_total_variance = sum(X_predictors .* X_predictors)

    # Return the CPPLS object containing all results
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
    gamma_optimization_tolerance::Real=1e-4 # -4
    ) where {T1<:Real, T2<:Real, T3<:Real}

    (X_predictors, Y_responses, Y_combined, observation_weights, X̄_mean, Ȳ_mean,
    X_deflated, X_loading_weights, X_loadings, Y_loadings, small_norm_flags, 
    regression_coefficients, _, _) = cppls_prepare_data(X_predictors, Y_responses, 
        n_components, Y_auxiliary, observation_weights, center)

    # Step 5: Iteratively compute components
    for i in 1:n_components
        # Compute optimal loading weights and canonical correlations, canonical 
        # coefficients, and if the power algorithm is applied, also optimal gamma values
        X_loading_weightsᵢ, _, _, _ = compute_cppls_weights(X_deflated, Y_combined, 
            Y_responses, observation_weights, gamma, gamma_optimization_tolerance)

        process_component!(i, X_deflated, X_loading_weightsᵢ, Y_responses,
            X_loading_weights, X_loadings, Y_loadings, regression_coefficients,
            small_norm_flags, X_tolerance, X_loading_weight_tolerance)

    end

    # Return the CPPLS object containing all results
    CPPLSLight(regression_coefficients, X̄_mean, Ȳ_mean)
end


# """
#     predict(cppls::CPPLS, X::AbstractMatrix{<:Real}) -> Array{Float64, 3}

# Generate predictions for a given input matrix `X` using a fitted CPPLS model.

# # Arguments
# - `cppls`: A fitted CPPLS model object containing regression coefficients, means, and other 
#   model parameters.
# - `X`: A matrix of predictor variables (observations × features) for which predictions are 
#   to be made.

# # Returns
# - A 3D array of size `(n_samples_X, n_targets_Y, n_components)`:
#   - `n_samples_X`: Number of rows in `X` (observations).
#   - `n_targets_Y`: Number of response variables in the CPPLS model.
#   - `n_components`: Number of components in the CPPLS model.
#   Each slice along the third dimension corresponds to the predictions for a specific number
#   of components.

# # Example
# ```julia
# # Example input data
# X = [
#     0.5  1.2  0.8;
#     1.0  0.9  1.1;
#     0.7  1.5  0.6
# ]

# # One-hot encoded response variable (Y)
# Y = [
#     1  0;
#     0  1;
#     1  0
# ]

# # Fit a CPPLS model (example)
# cppls = fit_cppls(X, Y, 2)

# # Generate predictions
# predictions = predict(cppls, X)
# 3×2×2 Array{Float64, 3}:
# [:, :, 1] =
#  0.86108     0.13892
#  0.0317869   0.968213
#  1.10713    -0.107134

# [:, :, 2] =
#  1.0  -5.55112e-17
#  0.0   1.0
#  1.0  -2.22045e-16

# # Output:
# # A 3D array of predictions with dimensions (n_samples_X, n_targets_Y, n_components)
# """
function predict(
    cppls::AbstractCPPLS,
    X::AbstractMatrix{<:Real},
    n_components::T=size(cppls.regression_coefficients, 3)) where T<:Integer

    # Get dimensions of input data and CPPLS model
    n_samples_X, _ = size(X)
    n_targets_Y = size(cppls.Y_means, 2)

    if n_components > size(cppls.regression_coefficients, 3)
        throw(DimensionMismatch(
            "n_components exceeds the number of components in the model"))
    end

    # Center the input data using the means from the CPPLS model
    X_centered = X .- cppls.X_means

    # Preallocate array for fitted values (predictions)
    fitted_values = similar(X, n_samples_X, n_targets_Y, n_components)

    # Compute predictions for each component
    for i in 1:n_components
        @views fitted_values[:, :, i] .= (X_centered * 
            cppls.regression_coefficients[:, :, i] .+ cppls.Y_means)
    end

    # Return the array of predictions
    fitted_values
end



# function one_hot_argmax(predictions::AbstractArray{<:Real, 3})
#     n_samples_X, n_targets_Y, _ = size(predictions)
    
#     # Sum over the last dimension to collapse it
#     summed_predictions = dropdims(sum(predictions, dims=3), dims=3)
    
#     # Get the predicted classes by finding the index of the maximum value in each row
#     max_indices = argmax(summed_predictions, dims=2)
    
#     # Convert CartesianIndex array to a 1D array of class indices
#     predicted_classes = getindex.(max_indices, 2)
    
#     # Create a one-hot encoded matrix
#     one_hot_encoded = zeros(Int, n_samples_X, n_targets_Y)
#     for sample_idx in 1:n_samples_X
#         one_hot_encoded[sample_idx, predicted_classes[sample_idx]] = 1
#     end
    
#     one_hot_encoded
# end


"""
    one_hot_argmax(predictions::Array{<:Real, 3}) -> Matrix{Int}

Convert a 3D array of predictions into a one-hot encoded 2D matrix based on the `argmax` 
along the second dimension.

# Arguments
- `predictions`: A 3D array of predictions with dimensions `(n_samples_X, n_targets_Y, 
  n_components)`. Typically, this represents predicted values for multiple samples, targets,
  and components.

# Returns
- A 2D matrix of size `(n_samples_X, n_targets_Y)` where each row is a one-hot encoded 
vector indicating the target class with the highest summed prediction across components.

# Example
```julia
predictions = reshape([0.1, 0.3, 0.2, 0.4, 0.7, 0.3, 0.5, 0.2, 0.1, 0.6, 0.4, 0.2], 2, 3, 2)
2×3×2 Array{Float64, 3}:
[:, :, 1] =
 0.1  0.2  0.7
 0.3  0.4  0.3

[:, :, 2] =
 0.5  0.1  0.4
 0.2  0.6  0.2

result = one_hot_argmax(predictions)
# Output:
# 2×3 Matrix{Int}:
# 0  0  1
# 0  1  0
"""
function one_hot_argmax(predictions::AbstractArray{<:Real, 3})
    # Sum over the last dimension to collapse it
    summed_predictions = dropdims(sum(predictions, dims=3), dims=3)
    n_labels = size(summed_predictions, 2)

    # Get the predicted classes by finding the index of the maximum value in each row
    predicted_classes = argmax(summed_predictions, dims=2)
    label_indices = vec(getindex.(predicted_classes, 2))  # Convert CartesianIndex to integers

    # Use labels_to_one_hot to convert class labels to a one-hot encoded matrix
    labels_to_one_hot(label_indices, n_labels)
end


function calculate_scores(cppls::AbstractCPPLS, X::AbstractMatrix{<:Real})
    # Ensure X is a Float64 matrix
    X = convert(Matrix{Float64}, X)  

    # Center the input data
    X = X .- cppls.X_means

    # Initialize the scores array
    X * cppls.projection
end