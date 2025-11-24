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

    strata_groups = Dict(stratum => findall(==(stratum), strata) 
        for stratum in unique(strata))

    batches = [Int[] for _ in 1:num_batches]

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

    class_labels = one_hot_to_labels(Y_train_full)
    inner_folds = random_batch_indices(class_labels, num_inner_folds, rng)
    
    best_num_latent_vars_per_fold = Vector{Int}(undef, num_inner_folds)

    for inner_fold_idx in 1:num_inner_folds_repeats

        test_indices = inner_folds[inner_fold_idx]

        verbose && println("  Inner fold: ", inner_fold_idx, " / ", num_inner_folds)

        @views X_validation = X_train_full[test_indices, :]
        @views Y_validation = Y_train_full[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_train_full[train_indices, :]
        @views Y_train = Y_train_full[train_indices, :]

        Y_auxiliary_train = Y_auxiliary !== nothing ? Y_auxiliary[train_indices, :] : Y_auxiliary

        misclassification_costs = Vector{Float64}(undef, max_components)

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
            Y_pred = predictonehot(model, predict(model, X_validation, num_components))
            misclassification_costs[num_components_idx] = nmc(Y_validation, Y_pred, 
                weighted_nmc)
        end

        best_num_latent_vars_per_fold[inner_fold_idx] = argmin(misclassification_costs)
        verbose && println("    Best number of latent variables in fold ", inner_fold_idx, ": ", 
            best_num_latent_vars_per_fold[inner_fold_idx])
    end

    best_num_latent_vars = floor(Int, median(best_num_latent_vars_per_fold))
    verbose && println("Best number of latent variables across folds: ", 
        best_num_latent_vars)
    best_num_latent_vars
end


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

    n_samples = size(X_predictors, 1)
    class_labels = one_hot_to_labels(Y_responses)
    outer_folds = random_batch_indices(class_labels, num_outer_folds, rng)
    
    outer_fold_accuracies = Vector{Float64}(undef, num_outer_folds_repeats)
    optimal_num_latent_variables = Vector{Int}(undef, num_outer_folds_repeats)

    for outer_fold_idx in 1:num_outer_folds_repeats

        test_indices = outer_folds[outer_fold_idx]
        
        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)
        
        @views X_test = X_predictors[test_indices, :]
        @views Y_test = Y_responses[test_indices, :]

        train_indices = setdiff(1:n_samples, test_indices)
        @views X_train = X_predictors[train_indices, :]
        @views Y_train = Y_responses[train_indices, :]

        Y_auxiliary_train = Y_auxiliary !== nothing ? Y_auxiliary[train_indices, :] : Y_auxiliary

        optimal_num_latent_variables[outer_fold_idx] = optimize_num_latent_variables(
            X_train, Y_train, max_components, num_inner_folds, num_inner_folds_repeats, 
            gamma, observation_weights, Y_auxiliary_train, center, X_tolerance, 
            X_loading_weight_tolerance, t_squared_norm_tolerance, 
            gamma_optimization_tolerance, weighted_nmc, rng, verbose)

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

        predicted_labels = predictonehot(final_model, predict(final_model, X_test))

        outer_fold_accuracies[outer_fold_idx] = 1 - nmc(predicted_labels, Y_test, weighted_nmc)

        verbose && println("Accuracy for outer fold: ", 
            outer_fold_accuracies[outer_fold_idx], "\n")
    end

    outer_fold_accuracies, optimal_num_latent_variables
end


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

        shuffled_indices = shuffle(1:n_samples)
        shuffled_Y_responses = @view Y_responses[shuffled_indices, :]

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

        permutation_accuracies[i] = mean(outer_fold_accuracies)
    end
    permutation_accuracies
end
