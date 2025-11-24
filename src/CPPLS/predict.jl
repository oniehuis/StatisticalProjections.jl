"""
    predict(
        cppls::CPPLS,
        X::AbstractMatrix{<:Real},
        n_components::Integer = size(cppls.regression_coefficients, 3)
    ) -> Array{Float64, 3}

Generate predictions from a fitted CPPLS model for a given input matrix `X`.

# Arguments
- `cppls`: A fitted CPPLS model, containing regression coefficients, mean values of 
  predictors and responses, etc.
- `X`: A matrix of predictor variables of size `(n_samples_X, n_features)`.
- `n_components` (optional): Number of CPPLS components to use for prediction. Defaults to 
  the full number trained in the model. Must not exceed the number available.

# Returns
- A 3-dimensional array of shape `(n_samples_X, n_targets_Y, n_components)`:
  - `n_samples_X`: Number of input samples (rows of `X`)
  - `n_targets_Y`: Number of target variables in the CPPLS model
  - `n_components`: Number of components used for prediction
  Each `[:,:,i]` slice corresponds to predictions using the first `i` components.

# Example
```julia
```
"""
function predict(
    cppls::AbstractCPPLS,
    X::AbstractMatrix{<:Real},
    n_components::T=size(cppls.regression_coefficients, 3)) where T<:Integer

    n_samples_X = size(X, 1)
    n_targets_Y = size(cppls.Y_means, 2)

    if n_components > size(cppls.regression_coefficients, 3)
        throw(DimensionMismatch(
            "n_components exceeds the number of components in the model"))
    end

    X_centered = X .- cppls.X_means
    fitted_values = similar(X, n_samples_X, n_targets_Y, n_components)

    for i in 1:n_components
        @views fitted_values[:, :, i] .= 
            (X_centered * cppls.regression_coefficients[:, :, i] .+ cppls.Y_means)
    end

    fitted_values
end

"""
    predictonehot(cppls::AbstractCPPLS, predictions::AbstractArray{<:Real, 3}) -> Matrix{Int}

Convert a 3D array of predictions from a CPPLS model into a one-hot encoded 2D matrix, 
assigning each sample to the class with the highest summed prediction across components, 
after adjusting for overcounted means.

# Arguments
- `cppls`: A fitted CPPLS model object containing the mean response vector (`Y_means`).
- `predictions`: A 3D array of predictions with dimensions `(n_samples_X, n_targets_Y, 
  n_components)`. 
  Typically, this represents predicted values for multiple samples, targets, and components.

# Returns
- A 2D matrix of size `(n_samples_X, n_targets_Y)` where each row is a one-hot encoded 
  vector indicating the target class with the highest summed prediction across components.

# Details
- Sums predictions across all components for each sample and class.
- Adjusts the summed predictions by subtracting `(n_components - 1)` times the mean 
  response, to correct for repeated addition of the mean in each component.
- For each sample, finds the class index with the highest adjusted prediction.
- Converts the predicted class indices to a one-hot encoded matrix.

# Example
```julia
```
"""
function predictonehot(cppls::AbstractCPPLS, predictions::AbstractArray{<:Real, 3})
    n_components = size(predictions, 3)
    n_classes = size(predictions, 2)

    Y_pred_sum = sum(predictions, dims=3)[:, :, 1]
    Y_pred_final = Y_pred_sum .- (n_components - 1) .* cppls.Y_means

    predicted_class_indices = argmax.(eachrow(Y_pred_final))

    labels_to_one_hot(predicted_class_indices, n_classes)
end

"""
    project(cppls::AbstractCPPLS, X::AbstractMatrix{<:Real}) -> AbstractMatrix

Project input data onto the latent components of a fitted CPPLS model to compute scores.

# Arguments
- `cppls`: A fitted CPPLS model object containing the projection matrix (`projection`) and 
  the mean of the predictor variables (`X_means`).
- `X`: A matrix of predictor variables with dimensions `(n_samples_X, n_features)`. These 
  should match the feature structure used to train the model.

# Returns
- A matrix of size `(n_samples_X, n_components)`, where each row contains the latent scores 
  for the corresponding sample in `X`.

# Details
- Centers the input matrix by subtracting the training mean `X_means`.
- Multiplies the centered data by the model's projection matrix to obtain component scores.

Example
```julia
```
"""
project(cppls::AbstractCPPLS, X::AbstractMatrix{<:Real}) = 
    (X .- cppls.X_means) * cppls.projection
