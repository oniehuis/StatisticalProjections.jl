"""
    clr_transform(matrix::AbstractMatrix{<:Real}; dims::Integer=2)

Perform the Centered Log-Ratio (CLR) transformation on a matrix.

# Arguments
- `matrix::AbstractMatrix{<:Real}`: The input matrix with positive values.
- `dims::Integer=2`: The dimension along which to compute the geometric mean (default is 2, 
rows).

# Returns
- A matrix with the CLR-transformed values.

# Raises
- `ArgumentError`: If the matrix contains values ≤ 0.
"""
function clr_transform(matrix::AbstractMatrix{<:Real}; dims::Integer=2)
    if any(matrix .<= 0)
        throw(ArgumentError(
            "The matrix must not contain values ≤ 0 for the CLR transformation."))
    end

    # Compute the geometric mean of each row or column
    geometric_means = exp.(mean(log.(matrix), dims=dims))

    # Perform the CLR transformation
    log.(matrix ./ geometric_means)
end


"""
    clr_transform!(matrix::AbstractMatrix{<:Real}; dims::Integer=2)

Perform the Centered Log-Ratio (CLR) transformation on a matrix in-place.

# Arguments
- `matrix::AbstractMatrix{<:Real}`: The input matrix with positive values.
- `dims::Integer=2`: The dimension along which to compute the geometric mean (default is 2, 
rows).

# Raises
- `ArgumentError`: If the matrix contains values ≤ 0.
"""
function clr_transform!(matrix::AbstractMatrix{<:Real}; dims::Integer=2)
    if any(matrix .<= 0)
        throw(ArgumentError(
            "The matrix must not contain values ≤ 0 for the CLR transformation."))
    end

    # Compute the geometric mean of each row or column
    geometric_means = exp.(mean(log.(matrix), dims=dims))

    # Perform the CLR transformation in-place
    @. matrix = log(matrix / geometric_means)
end


function labels_to_one_hot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)
    n_samples = length(label_indices)
    one_hot = zeros(Int, n_samples, n_labels)
    for i in 1:n_samples
        one_hot[i, label_indices[i]] = 1
    end
    one_hot
end


function labels_to_one_hot(labels::AbstractVector)
    unique_labels = sort(collect(Set(labels)))  # consistent label order
    label_to_index = Dict(label => i for (i, label) in enumerate(unique_labels))
    
    num_classes = length(unique_labels)
    num_samples = length(labels)
    one_hot = zeros(Int, num_samples, num_classes)
    
    for (i, label) in enumerate(labels)
        idx = label_to_index[label]
        one_hot[i, idx] = 1
    end
    
    one_hot, unique_labels
end