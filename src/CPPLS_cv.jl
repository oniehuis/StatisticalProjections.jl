using JLD2

function random_batch_indices(
    strata::AbstractVector{<:Integer},
    num_batches::Integer,
    rng::AbstractRNG=Random.GLOBAL_RNG)

    n_samples = length(strata)

    if num_batches < 1
        throw(ArgumentError("Number of batches must be at least 1."))
    end
    if num_batches > n_samples
        throw(ArgumentError(
            "Number of batches ($num_batches) exceeds number of samples ($n_samples)."))
    end

    # Group indices by strata
    strata_groups = Dict(stratum => findall(==(stratum), strata) 
        for stratum in unique(strata))

    # Initialize k empty batches
    batches = [Int[] for _ in 1:num_batches]

    # Distribute elements from each stratum round-robin
    for (stratum, indices) in strata_groups
        shuffled = shuffle(rng, indices)
        n = length(shuffled)
        if !(n % num_batches ≈ 0)
            @info ("Stratum $stratum (size = $n) not evenly divisible by " 
                * "$num_batches batches.")
        end
        for (i, idx) in enumerate(shuffled)
            push!(batches[mod1(i, num_batches)], idx)
        end
    end

    batches
end


function one_hot_to_labels(one_hot_matrix::AbstractMatrix{<:Integer})
    [argmax(row) for row in eachrow(one_hot_matrix)]
end


function nmc(
    Y_true_one_hot::AbstractMatrix{<:Integer}, 
    Y_pred_one_hot::AbstractMatrix{<:Integer},
    weighted::Bool)

    # Check input dimensions
    size(Y_true_one_hot) == size(Y_pred_one_hot) || 
        throw(DimensionMismatch("Input matrices must have the same dimensions"))

    n_samples = size(Y_true_one_hot, 1)
    n_samples > 0 || error("Cannot compute weighted NMC: input has zero samples")

    !weighted && return mean(Y_true_one_hot .≠ Y_pred_one_hot)

    # Convert one-hot to class labels using argmax per row
    true_labels = one_hot_to_labels(Y_true_one_hot)
    pred_labels = one_hot_to_labels(Y_pred_one_hot)

    # Compute normalized inverse-frequency weights
    class_counts = countmap(true_labels)
    inv_freqs = Dict(k => n_samples / v for (k, v) in class_counts)
    total_weight = sum(inv_freqs[k] * class_counts[k] for k in keys(inv_freqs))
    class_weights = Dict(k => inv_freqs[k] / total_weight for k in keys(inv_freqs))

    # Get per-sample weights and error mask
    sample_weights = getindex.(Ref(class_weights), true_labels)
    errors = true_labels .≠ pred_labels

    # Compute final weighted error
    weighted_error = sum(sample_weights[errors])
    clamp(weighted_error, 0.0, 1.0)
end


function optimize_num_latent_variables(
    X_train_full::AbstractMatrix{<:Real}, 
    Y_train_full::AbstractMatrix{<:Integer}, 
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    gamma::Union{<:Real, <:NTuple{2, <:Real}, <:AbstractVector{<:Union{<:Real, <:NTuple{2, <:Real}}}}, 
    observation_weights::Union{AbstractVector{<:Real}, Nothing},
    Y_auxiliary::Union{AbstractMatrix{<:Real}, Nothing},
    center::Bool,
    X_tolerance::Real,
    X_loading_weight_tolerance::Real,
    t_squared_norm_tolerance::Real,
    gamma_optimization_tolerance::Real,
    weighted_nmc::Bool,
    rng::AbstractRNG,
    verbose::Bool)
    
    n_samples = size(X_train_full, 1)

    # Generate inner cross-validation folds
    class_labels = one_hot_to_labels(Y_train_full)
    inner_folds = random_batch_indices(class_labels, num_inner_folds, rng)
    
    # Preallocate vector to store the best number of latent variables for each fold
    best_num_latent_vars_per_fold = Vector{Int}(undef, num_inner_folds)

    for inner_fold_idx in 1:num_inner_folds_repeats

        test_indices = inner_folds[inner_fold_idx]

        verbose && println("  Inner fold: ", inner_fold_idx, " / ", num_inner_folds)

        # Split data into training and validation sets for the current inner fold
        @views X_validation = X_train_full[test_indices, :]
        @views Y_validation = Y_train_full[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_train_full[train_indices, :]
        @views Y_train = Y_train_full[train_indices, :]

        # If Y_auxiliary is provided, ensure it has the same number of rows as X_train
        Y_auxiliary_train = Y_auxiliary !== nothing ? Y_auxiliary[train_indices, :] : Y_auxiliary

        # Preallocate vector to store misclassification costs for each latent variable
        misclassification_costs = Vector{Float64}(undef, max_components)

        # Train model and evaluate for each number of latent variables
        model = fit_cppls_light(X_train, Y_train, max_components, 
            gamma=gamma,
            observation_weights=observation_weights,
            Y_auxiliary=Y_auxiliary_train,
            center=center,
            X_tolerance=X_tolerance, 
            X_loading_weight_tolerance=X_loading_weight_tolerance,
            t_squared_norm_tolerance=t_squared_norm_tolerance,
            gamma_optimization_tolerance=gamma_optimization_tolerance)

        for (num_components_idx, num_components) in enumerate(1:max_components)
            Y_pred = one_hot_argmax(predict(model, X_validation, num_components))
            misclassification_costs[num_components_idx] = nmc(Y_validation, Y_pred, 
                weighted_nmc)
        end

        # Select the number of latent variables with the lowest misclassification cost
        best_num_latent_vars_per_fold[inner_fold_idx] = argmin(misclassification_costs)
        verbose && println("    Best number of latent variables in fold ", inner_fold_idx, ": ", 
            best_num_latent_vars_per_fold[inner_fold_idx])
    end

    # Return the median of the best latent variables across all folds
    best_num_latent_vars = floor(Int, median(best_num_latent_vars_per_fold))
    verbose && println("Best number of latent variables across folds: ", 
        best_num_latent_vars)
    best_num_latent_vars
end


"""
    nested_cv(X_predictors::AbstractMatrix{<:Real}, 
              Y_responses::AbstractMatrix{<:Real};
              gamma::Union{<:T1, <:NTuple{2, T1}, <:AbstractVector{Union{<:T1, 
                <:NTuple{2, T1}}}}=0.5,
              observation_weights::Union{AbstractVector{T2}, Nothing}=nothing,
              Y_auxiliary::Union{AbstractMatrix{T3}, Nothing}=nothing,
              center::Bool=true,
              X_tolerance::Real=1e-12,
              X_loading_weight_tolerance::Real=eps(Float64),
              gamma_optimization_tolerance::Real=1e-4,
              num_outer_folds::Integer=8,
              num_outer_folds_repeats::Integer=num_outer_folds,
              num_inner_folds::Integer=7,
              num_inner_folds_repeats::Integer=num_inner_folds,
              max_components::Integer=5,
              weighted_nmc::Bool=true,
              rng::AbstractRNG=Random.GLOBAL_RNG,
              verbose::Bool=true) 
              -> Tuple{Vector{Float64}, Vector{Int}}

Perform nested cross-validation to evaluate the predictive performance of a CPPLS-DA model 
and determine the optimal number of latent variables.

# Arguments
- `X_predictors`: Predictor matrix (samples × features).
- `Y_responses`: Response matrix (samples × classes or targets), assumed to be one-hot 
    encoded.
- `gamma`: Regularization parameter(s). Can be a scalar, a tuple for range search, or a 
    vector of values.
- `observation_weights`: Optional weights for training samples.
- `Y_auxiliary`: Optional auxiliary response matrix.
- `center`: Whether to mean-center the predictors.
- `X_tolerance`: Tolerance for convergence in predictor matrix.
- `X_loading_weight_tolerance`: Tolerance for convergence of loading weights.
- `gamma_optimization_tolerance`: Tolerance for `gamma` optimization.
- `num_outer_folds`: Number of outer CV folds.
- `num_outer_folds_repeats`: Number of times to repeat outer CV.
- `num_inner_folds`: Number of inner CV folds used to estimate the number of latent 
   variables/components.
- `num_inner_folds_repeats`: Number of repetitions of inner CV.
- `max_components`: Maximum number of latent variables to search over.
- `weighted_nmc`: Whether to use class-weighted normalized misclassification.
- `rng`: Random number generator used for reproducibility.
- `verbose`: Whether to print progress messages.

# Returns
- `outer_fold_accuracies`: A vector of classification accuracies (or 1 - NMC) for each 
    outer fold.
- `optimal_num_latent_variables`: A vector of the selected number of latent variables per 
    outer fold.

# Description
Nested cross-validation involves:
1. **Outer folds** for evaluating generalization error.
2. **Inner folds** for tuning hyperparameters (e.g., number of latent variables).

In each outer fold:
- The model is trained and optimized on the training portion (using inner folds).
- It is evaluated on the held-out test data using classification accuracy (via `nmc`).

# Example
```julia
X = rand(100, 10)
Y = labels_to_one_hot(rand(1:3, 100), 3)  # one-hot encoding for 3-class classification

accs, opt_lv = nested_cv(
    X, Y;
    gamma=0.5,
    num_outer_folds=5,
    num_inner_folds=3,
    max_components=10,
    verbose=true
)

println("Accuracies: ", accs)
println("Optimal latent variables: ", opt_lv)
```
"""
function nested_cv(
    X_predictors::AbstractMatrix{<:Real}, 
    Y_responses::AbstractMatrix{<:Real};
    gamma::Union{<:T1, <:NTuple{2, T1}, <:AbstractVector{Union{<:T1, <:NTuple{2, T1}}}}=0.5,
    observation_weights::Union{AbstractVector{T2}, Nothing}=nothing,
    Y_auxiliary::Union{AbstractMatrix{T3}, Nothing}=nothing,
    center::Bool=true,
    X_tolerance::Real=1e-12,
    X_loading_weight_tolerance::Real=eps(Float64),
    t_squared_norm_tolerance::Real=1e-10,
    gamma_optimization_tolerance::Real=1e-4,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    max_components::Integer=5,
    weighted_nmc::Bool=true,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true
    ) where {T1<:Real, T2<:Real, T3<:Real}
    
    num_outer_folds_repeats ≤ num_outer_folds ||
        error("The number of outer fold repeats cannot exceed the number of outer folds")

    num_inner_folds_repeats ≤ num_inner_folds ||
        error("The number of inner fold repeats cannot exceed the number of inner folds")

    max_components > 0 ||
        error("The number of components must be greater than zero")

    # Validate input parameters
    n_samples = size(X_predictors, 1)
    
    # Convert response matrix to class labels
    class_labels = one_hot_to_labels(Y_responses)
    
    # Generate outer cross-validation folds
    outer_folds = random_batch_indices(class_labels, num_outer_folds, rng)
    
    # Preallocate vector to store accuracies for each outer fold
    outer_fold_accuracies = Vector{Float64}(undef, num_outer_folds_repeats)
    optimal_num_latent_variables = Vector{Int}(undef, num_outer_folds_repeats)

    for outer_fold_idx in 1:num_outer_folds_repeats

        # outer_fold_idx > num_outer_folds_repeats && break
        test_indices = outer_folds[outer_fold_idx]
        
        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)
        
        # Split data into test and training sets for the current outer fold
        @views X_test = X_predictors[test_indices, :]
        @views Y_test = Y_responses[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_predictors[train_indices, :]
        @views Y_train = Y_responses[train_indices, :]

        # If Y_auxiliary is provided, ensure it has the same number of rows as X_train
        Y_auxiliary_train = Y_auxiliary !== nothing ? Y_auxiliary[train_indices, :] : Y_auxiliary

        # Run inner cross-validation to find the optimal number of latent variables
        optimal_num_latent_variables[outer_fold_idx] = optimize_num_latent_variables(
            X_train, Y_train, max_components, num_inner_folds, num_inner_folds_repeats, 
            gamma, observation_weights, Y_auxiliary_train, center, X_tolerance, 
            X_loading_weight_tolerance, t_squared_norm_tolerance, 
            gamma_optimization_tolerance, weighted_nmc, rng, verbose)

        # Train the final model using the optimal number of latent variables
        final_model = fit_cppls_light(
            X_train,
            Y_train, 
            optimal_num_latent_variables[outer_fold_idx], 
            gamma=gamma,
            observation_weights=observation_weights,
            Y_auxiliary=Y_auxiliary_train,
            center=center,
            X_tolerance=X_tolerance, 
            X_loading_weight_tolerance=X_loading_weight_tolerance,
            t_squared_norm_tolerance=t_squared_norm_tolerance,
            gamma_optimization_tolerance=gamma_optimization_tolerance)

        # Predict the response labels for the test set
        predicted_labels = one_hot_argmax(predict(final_model, X_test))

        # Compute accuracy for the current outer fold
        outer_fold_accuracies[outer_fold_idx] = 1 - nmc(predicted_labels, Y_test, weighted_nmc)

        verbose && println("Accuracy for outer fold: ", 
            outer_fold_accuracies[outer_fold_idx], "\n")
    end

    # Return the accuracies and the optimal number of latent variables for all outer folds
    outer_fold_accuracies, optimal_num_latent_variables
end


"""
    nested_cv_permutation(X_predictors::AbstractMatrix{<:Real}, 
                          Y_responses::AbstractMatrix{<:Real};
                          gamma::Union{<:T1, <:NTuple{2, T1}, <:AbstractVector{<:Union{<:T1, 
                            <:NTuple{2, T1}}}}=0.5,
                          observation_weights::Union{AbstractVector{T2}, Nothing}=nothing,
                          Y_auxiliary::Union{AbstractMatrix{T3}, Nothing}=nothing,
                          center::Bool=true,
                          X_tolerance::Real=1e-12,
                          X_loading_weight_tolerance::Real=eps(Float64),
                          gamma_optimization_tolerance::Real=1e-4,
                          num_outer_folds::Integer=9,
                          num_outer_folds_repeats::Integer=num_outer_folds,
                          num_inner_folds::Integer=8,
                          num_inner_folds_repeats::Integer=num_inner_folds,
                          max_components::Integer=5,
                          weighted_nmc::Bool=true,
                          num_permutations::Integer=999,
                          rng::AbstractRNG=Random.GLOBAL_RNG,
                          verbose::Bool=true)
                          -> Vector{Float64}

Perform permutation testing using nested cross-validation to assess the statistical 
significance of a CPPLS-DA model’s predictive performance.

# Arguments
- `X_predictors`: Predictor matrix (samples × features).
- `Y_responses`: Response matrix (samples × targets), assumed one-hot encoded.
- `gamma`: Regularization parameter(s). Can be scalar, tuple, or array of values.
- `observation_weights`: Optional weights for training samples.
- `Y_auxiliary`: Optional auxiliary response matrix.
- `center`: Whether to mean-center the predictors.
- `X_tolerance`: Tolerance for convergence in predictor matrix.
- `X_loading_weight_tolerance`: Tolerance for convergence of loading weights.
- `gamma_optimization_tolerance`: Tolerance for `gamma` optimization.
- `num_outer_folds`: Number of outer CV folds.
- `num_outer_folds_repeats`: Number of times to repeat outer CV.
- `num_inner_folds`: Number of inner CV folds used to estimate the number of latent 
   variables/components.
- `num_inner_folds_repeats`: Number of repetitions of inner CV.
- `max_components`: Maximum number of latent variables to search over.
- `weighted_nmc`: Whether to use class-weighted normalized misclassification.
- `num_permutations`: Number of random permutations of the response labels.
- `rng`: Random number generator used for reproducibility.
- `verbose`: Whether to print progress during permutations.

# Returns
- `permutation_accuracies`: A vector of mean accuracies (or 1 - NMC) from each permutation 
   run.

# Description
This function performs a permutation test using nested cross-validation to build a null 
distribution of model performance. In each permutation:
- The response labels (`Y_responses`, `Y_auxiliary`) are randomly shuffled.
- Nested CV is performed with the permuted labels to compute a baseline performance under 
  the null hypothesis.
- This is repeated `num_permutations` times to yield a distribution of accuracies.

This is useful for estimating the probability of achieving the observed model performance 
by chance, thereby enabling statistical significance testing.

# Notes
- The input `Y_responses` must be one-hot encoded.
- This function relies on the `nested_cv` function for cross-validation logic.
- The output can be compared to actual model accuracy to compute p-values.

# Example
```julia
X = rand(100, 10)
Y = labels_to_one_hot(rand(1:3, 100), 3)

permutation_scores = nested_cv_permutation(
    X, Y;
    gamma=0.5,
    num_permutations=100,
    num_outer_folds=5,
    num_inner_folds=3,
    max_components=5,
    verbose=true
)

println("Mean permutation accuracy: ", mean(permutation_scores))
```
"""
function nested_cv_permutation(
    X_predictors::AbstractMatrix{<:Real}, 
    Y_responses::AbstractMatrix{<:Real};
    gamma::Union{<:T1, <:NTuple{2, T1}, <:AbstractVector{<:Union{<:T1, <:NTuple{2, T1}}}}=0.5,
    observation_weights::Union{AbstractVector{T2}, Nothing}=nothing,
    Y_auxiliary::Union{AbstractMatrix{T3}, Nothing}=nothing,
    center::Bool=true,
    X_tolerance::Real=1e-12,
    X_loading_weight_tolerance::Real=eps(Float64),
    t_squared_norm_tolerance::Real=1e-10,
    gamma_optimization_tolerance::Real=1e-4,
    num_outer_folds::Integer=9,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=8,
    num_inner_folds_repeats::Integer=num_inner_folds,
    max_components::Integer=5,
    weighted_nmc::Bool=true,
    num_permutations::Integer=999,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
    ) where {T1<:Real, T2<:Real, T3<:Real}

    num_outer_folds_repeats ≤ num_outer_folds ||
        error("The number of outer fold repeats cannot exceed the number of outer folds")

    num_inner_folds_repeats ≤ num_inner_folds ||
        error("The number of inner fold repeats cannot exceed the number of inner folds")

    max_components > 0 ||
        error("The number of components must be greater than zero")

    n_samples = size(X_predictors, 1)
    permutation_accuracies = Vector{Float64}(undef, num_permutations)

    for i in 1:num_permutations
        verbose && println("Permutation: ", i, " / ", num_permutations)

        # Shuffle the response labels
        shuffled_indices = shuffle(1:n_samples)
        shuffled_Y_responses = @view Y_responses[shuffled_indices, :]

        # Run nested cross-validation with the permuted labels
        outer_fold_accuracies, _ = nested_cv(
            X_predictors, 
            shuffled_Y_responses;
            gamma=gamma,
            observation_weights=observation_weights,
            Y_auxiliary=Y_auxiliary,
            center=center,
            X_tolerance=X_tolerance,
            X_loading_weight_tolerance=X_loading_weight_tolerance,
            t_squared_norm_tolerance=t_squared_norm_tolerance,
            gamma_optimization_tolerance=gamma_optimization_tolerance,
            num_outer_folds=num_outer_folds,
            num_outer_folds_repeats=num_outer_folds_repeats,
            num_inner_folds=num_inner_folds,
            num_inner_folds_repeats=num_inner_folds_repeats,
            max_components=max_components,
            weighted_nmc=weighted_nmc,
            rng=rng,
            verbose=verbose)

        permuted_accuracy = mean(outer_fold_accuracies)
        permutation_accuracies[i] = permuted_accuracy
    end
    permutation_accuracies
end


"""
    calculate_p_value(permutation_accuracies::AbstractVector{<:Real}, 
                      model_accuracy::Float64) -> Float64

Calculate the p-value to assess the statistical significance of the model's accuracy based 
on permutation testing.

# Arguments
- `permutation_accuracies::AbstractVector{<:Real}`: A vector of accuracies obtained from 
   models trained on permuted response labels.
- `model_accuracy::Float64`: The accuracy of the model trained on the original 
   (non-permuted) response labels.

# Returns
- `p_value::Float64`: The p-value representing the probability of observing the model's 
   accuracy (or better) by random chance.

# Description
The p-value is calculated as the proportion of permutation accuracies that are less than or 
equal to the model's accuracy. This provides a measure of how likely it is to achieve the 
observed accuracy by chance, under the null hypothesis that the response labels are 
unrelated to the predictors.

The formula for the p-value is:
p_value = (number of permutation accuracies ≤ model_accuracy) / 
    (total number of permutations + 1)

# Example
```julia
permutation_accuracies = [0.6, 0.55, 0.58, 0.62, 0.57]
model_accuracy = 0.7

p_value = calculate_p_value(permutation_accuracies, model_accuracy)
println("P-value: ", p_value)
"""
function calculate_p_value(
    permutation_accuracies::AbstractVector{<:Real}, 
    model_accuracy::Float64)

    (count(x -> x ≤ model_accuracy || x ≈ model_accuracy, permutation_accuracies) 
        / (length(permutation_accuracies) + 1))
end
