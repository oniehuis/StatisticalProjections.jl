"""
    labels_to_one_hot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)

Convert integer label indices (1-based) to a dense one-hot encoded matrix with
`n_labels` columns. This variant assumes the set of classes is already known and
returns only the encoded array.
"""
function labels_to_one_hot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)
    n_samples = length(label_indices)
    one_hot = zeros(Int, n_samples, n_labels)
    @inbounds for i in 1:n_samples
        one_hot[i, label_indices[i]] = 1
    end
    one_hot
end


"""
    labels_to_one_hot(labels::AbstractVector)

Encode arbitrary labels (e.g., strings, integers, symbols) into a one-hot matrix,
automatically determining the unique label ordering. Returns a tuple of the
encoded matrix and the ordered list of labels so callers can map predictions
back to the original domain.
"""
function labels_to_one_hot(labels::AbstractVector)
    unique_labels = sort(collect(Set(labels)))  # consistent label order
    label_to_index = Dict(label => i for (i, label) in enumerate(unique_labels))
    
    num_classes = length(unique_labels)
    num_samples = length(labels)
    one_hot = zeros(Int, num_samples, num_classes)
    
    @inbounds for (i, label) in enumerate(labels)
        idx = label_to_index[label]
        one_hot[i, idx] = 1
    end
    
    one_hot, unique_labels
end


"""
    one_hot_to_labels(one_hot_matrix::AbstractMatrix{<:Integer})

Decode one-hot rows back into label indices by selecting the column of the
maximum entry for each row. Works with any integer-valued matrix containing a
single positive entry per row.
"""
one_hot_to_labels(one_hot_matrix::AbstractMatrix{<:Integer}) =
    [argmax(row) for row in eachrow(one_hot_matrix)]
