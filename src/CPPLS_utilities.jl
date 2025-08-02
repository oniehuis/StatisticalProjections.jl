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


"""
    find_invariant_and_variant_columns(M::AbstractMatrix)

Identify invariant and variant columns in a matrix.

# Arguments
- `M::AbstractMatrix`: The input matrix where each column is analyzed to determine whether 
  it is invariant (all elements are the same) or variant (elements differ).

# Returns
- `(invariant_columns::Vector{Int}, variant_columns::Vector{Int})`: 
  A tuple containing:
  - `invariant_columns`: A vector of column indices where all elements are the same.
  - `variant_columns`: A vector of column indices where at least one element differs.

# Description
This function iterates over the columns of the input matrix `M` and checks whether all 
elements in each column are the same. Columns with identical elements are classified as 
"invariant," while columns with differing elements are classified as "variant."

# Example
```julia
julia> M = [1 2 2; 1 1 2; 1 2 2]
3×3 Matrix{Int64}:
 1  2  2
 1  1  2
 1  2  2

julia> invariant_columns, variant_columns = find_invariant_and_variant_columns(M)
([1, 3], [2])

"""
function find_invariant_and_variant_columns(M::AbstractMatrix)
    invariant_columns = Int[]
    variant_columns = Int[]

    for j in axes(M, 2)
        col = @view M[:, j]
        if all(x -> x == col[1], col)
            push!(invariant_columns, j)
        else
            push!(variant_columns, j)
        end
    end
    
    invariant_columns, variant_columns
end
