function nmc(
    Y_true_one_hot::AbstractMatrix{<:Integer}, 
    Y_pred_one_hot::AbstractMatrix{<:Integer},
    weighted::Bool)

    size(Y_true_one_hot) == size(Y_pred_one_hot) || 
        throw(DimensionMismatch("Input matrices must have the same dimensions"))

    n_samples = size(Y_true_one_hot, 1)
    n_samples > 0 || error("Cannot compute weighted NMC: input has zero samples")

    !weighted && return mean(Y_true_one_hot .≠ Y_pred_one_hot)

    true_labels = one_hot_to_labels(Y_true_one_hot)
    pred_labels = one_hot_to_labels(Y_pred_one_hot)

    class_counts = countmap(true_labels)
    inv_freqs = Dict(k => n_samples / v for (k, v) in class_counts)
    total_weight = sum(inv_freqs[k] * class_counts[k] for k in keys(inv_freqs))
    class_weights = Dict(k => inv_freqs[k] / total_weight for k in keys(inv_freqs))

    sample_weights = getindex.(Ref(class_weights), true_labels)
    errors = true_labels .≠ pred_labels

    weighted_error = sum(sample_weights[errors])
    clamp(weighted_error, 0.0, 1.0)
end


function calculate_p_value(
    permutation_accuracies::AbstractVector{<:Real}, 
    model_accuracy::Float64)

    (count(x -> x ≤ model_accuracy || x ≈ model_accuracy, permutation_accuracies) 
        / (length(permutation_accuracies) + 1))
end
