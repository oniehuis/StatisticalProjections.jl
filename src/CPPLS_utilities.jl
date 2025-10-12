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


# """
#     find_invariant_and_variant_columns(M::AbstractMatrix)

# Identify invariant and variant columns in a matrix.

# # Arguments
# - `M::AbstractMatrix`: The input matrix where each column is analyzed to determine whether 
#   it is invariant (all elements are the same) or variant (elements differ).

# # Returns
# - `(invariant_columns::Vector{Int}, variant_columns::Vector{Int})`: 
#   A tuple containing:
#   - `invariant_columns`: A vector of column indices where all elements are the same.
#   - `variant_columns`: A vector of column indices where at least one element differs.

# # Description
# This function iterates over the columns of the input matrix `M` and checks whether all 
# elements in each column are the same. Columns with identical elements are classified as 
# "invariant," while columns with differing elements are classified as "variant."

# # Example
# ```julia
# julia> M = [1 2 2; 1 1 2; 1 2 2]
# 3×3 Matrix{Int64}:
#  1  2  2
#  1  1  2
#  1  2  2

# julia> invariant_columns, variant_columns = find_invariant_and_variant_columns(M)
# ([1, 3], [2])

# """
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


# """
#     robustcor(x, y)

# Compute the Pearson correlation between two vectors, returning `0.0` in degenerate cases.

# # Arguments
# - `x::AbstractVector`
# - `y::AbstractVector`

# # Returns
# - `c::Float64` — the correlation coefficient in [-1, 1].

# # Behavior
# - If either vector has zero standard deviation, returns `0.0`.  
# - If `cor(x, y)` is `NaN` or `Inf`, returns `0.0`.  
# - Otherwise returns the finite correlation value.  

# # Example
# ```julia
# robustcor([1,2,3], [2,4,6])    # 1.0
# robustcor([1,1,1], [2,3,4])    # 0.0 (zero variance in x)
# robustcor([1,2,3], [NaN,2,3])  # 0.0 (invalid correlation)
# ```
# """
@inline function robustcor(x::AbstractVector, y::AbstractVector)
    (std(x) == 0 || std(y) == 0) && return 0.0
    c = cor(x, y)
    isfinite(c) ? c : 0.0
end

# """
#     fisherztrack(X, scores; weights=:mean)

# Compute a Fisher-z–averaged correlation track across the third dimension
# for each position along the second dimension of a 3D data array.

# # Arguments
# - `X::AbstractArray{<:Real,3}`  
#   Data cube of shape `(n_samples, n_axis₁, n_axis₂)`.  
#   The first dimension indexes samples, the second is the axis along which
#   correlations are reported, the third are the channels averaged with Fisher’s z.  

# - `scores::AbstractVector{<:Real}`  
#   Vector of length `n_samples`. Represents a single 1-D target signal
#   (e.g. class separation scores) to correlate against.  

# - `weights::Symbol = :mean`  
#   How to weight correlations across the third dimension:  
#   * `:mean` — weight by the mean intensity of each channel (default).  
#   * `:none` — equal weights for all channels.  

# # Returns
# - `ρ::Vector{Float64}` of length `n_axis₁`.  
#   Each entry is the Fisher-z–averaged correlation between `scores` and
#   all channels at that position of axis₁.

# # Notes
# - Correlations are clamped to the open interval (-1, 1) before Fisher’s
#   transform to avoid infinities.  
# - Fisher’s z stabilizes correlations before averaging; the result is then
#   back-transformed to a standard correlation coefficient.  

# # Example
# ```julia
# X = randn(20, 50, 10)        # 20 samples × 50 axis₁ × 10 axis₂
# scores = randn(20)           # separation scores per sample
# ρ = fisherztrack(X, scores)  # length 50 correlation track
# """
function fisherztrack(X::AbstractArray{<:Real,3},
                      scores::AbstractVector{<:Real};
                      weights::Symbol=:mean)

    # Input checks
    n_samples, n_axis₁, n_axis₂ = size(X)
    length(scores) == n_samples ||
        throw(ArgumentError("scores length must equal size(X, 1)."))
    (weights === :mean || weights === :none) ||
        throw(ArgumentError("weights must be :mean or :none."))

    # Output: one correlation value per axis₁ position
    ρ = Vector{Float64}(undef, n_axis₁)

    # Numerical safety: Fisher transform requires |r| < 1
    lo = nextfloat(-1.0)   # just above -1
    hi = prevfloat(1.0)    # just below +1

    @inbounds for a₁ in 1:n_axis₁
        # Collect correlations along axis₂ for this axis₁ position
        rs = Vector{Float64}(undef, n_axis₂)
        ws = ones(Float64, n_axis₂)

        for a₂ in 1:n_axis₂
            xs = @view X[:, a₁, a₂]   # sample vector for this axis₂
            rs[a₂] = robustcor(xs, scores)
            if weights === :mean
                ws[a₂] = mean(xs)
            end
        end

        # Fisher z-transform → weighted mean in z-space → back-transform
        zs = atanh.(clamp.(rs, lo, hi))
        z̄ = sum(ws .* zs) / (sum(ws) + eps(Float64))
        ρ[a₁] = tanh(z̄)
    end

    ρ
end

# """
#     separationaxis(Xscores, Y; method=:centroid, positive_class=1)

# Compute a one-dimensional separation axis in score space for a binary problem.  

# # Arguments
# - `Xscores::AbstractMatrix{<:Real}`  
#   An `n × p` score matrix (rows = samples, columns = score axes).  
#   * `p = 1`: Fast path — the axis is the single score dimension.  
#   * `p ≥ 2`: Axis is computed by centroid or LDA method.  

# - `Y::AbstractMatrix{<:Real}`  
#   An `n × 2` strict one-hot class indicator matrix. Each row must have
#   exactly one entry equal to 1, the other 0. Both classes must be present.  

# - `method::Symbol = :centroid`  
#   How to compute the separation axis (only used when `p ≥ 2`):  
#   * `:centroid` — Difference between class centroids.  
#   * `:lda` — Fisher’s linear discriminant: inverse pooled covariance × centroid difference.  

# - `positive_class::Integer = 1`  
#   Ensures the returned axis is oriented so that this class has the higher mean score.  

# # Returns
# A tuple `(direction, scores)` where:  
# - `direction :: Vector{<:Real}` — Normalized separation axis in score space  
#   (`[±1]` if `p == 1`).  
# - `scores :: Vector{<:Real}` — Projection of each sample onto the axis.  

# # Notes
# - If the computed axis is zero or invalid, an error is thrown.  
# - When `p == 1`, the separation axis is trivial; only orientation is applied.  

# # Example
# ```julia
# X = randn(100, 2)                 # scores for 100 samples, 2 components
# Y = [rand(Bool) for _ in 1:100]   # random labels
# Y = hcat(Int.(Y), 1 .- Int.(Y))   # one-hot 100 × 2

# direction, scores = separationaxis(X, Y; method=:lda, positive_class=2)
# ```
# """
function separationaxis(Xscores::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
                        method::Symbol=:centroid, positive_class::T=1) where {T<:Integer}

    # Validate input dimensions
    n, p = size(Xscores)
    p ≥ 1 || throw(ArgumentError("Expecting at least one score axis (got $p)."))
    size(Y, 1) == n || throw(ArgumentError("Row count mismatch between Xscores and Y."))
    size(Y, 2) == 2 || throw(ArgumentError("Binary only; for K > 2 use one-vs-rest."))

    # Validate Y as strict one-hot
    all((Y .== 0) .| (Y .== 1)) ||
        throw(ArgumentError("Y must be strictly one-hot (0/1)."))
    all(sum(Y, dims=2) .== 1) ||
        throw(ArgumentError("Each row of Y must have exactly one '1'."))

    # Indices for each class, both must be non-empty
    idx₁ = findall(Y[:, 1] .== 1)
    idx₂ = findall(Y[:, 2] .== 1)
    (!isempty(idx₁) && !isempty(idx₂)) ||
        throw(ArgumentError("Both classes must be present."))

    if p == 1
        # Single-axis shortcut: direction = [±1], scores = the axis itself
        s = vec(@view Xscores[:, 1])
        direction = ones(eltype(Xscores), 1)
        scores = copy(s)

        # Orient so positive_class has higher mean
        m₁, m₂ = mean(scores[idx₁]), mean(scores[idx₂])
        want   = positive_class == 1 ? m₁ : m₂
        other  = positive_class == 1 ? m₂ : m₁
        if want < other
            direction .= -direction
            scores    .= -scores
        end
        
        return direction, scores

    else
        # Compute centroids
        X₁ = @view Xscores[idx₁, :]
        X₂ = @view Xscores[idx₂, :]
        μ₁ = vec(mean(X₁, dims=1))
        μ₂ = vec(mean(X₂, dims=1))

        # Separation direction from centroid diff or LDA
        direction = if method === :centroid
            μ₁ - μ₂
        elseif method === :lda
            S₁ = cov(X₁; corrected=true)
            S₂ = cov(X₂; corrected=true)
            (S₁ + S₂) \ (μ₁ - μ₂)
        else
            throw(ArgumentError("method must be :centroid or :lda"))
        end

        # Guard against zero or invalid direction
        if !any(isfinite, direction) || norm(direction) == 0
            throw(ArgumentError("Separation axis is undefined (zero vector). Check class means."))
        end

        # Normalize and project
        direction ./= (norm(direction) + eps(eltype(direction)))
        scores = Xscores * direction

        # Orient so positive_class has higher mean
        m₁, m₂ = mean(scores[idx₁]), mean(scores[idx₂])
        want   = positive_class == 1 ? m₁ : m₂
        other  = positive_class == 1 ? m₂ : m₁
        if want < other
            direction .= -direction
            scores    .= -scores
        end

        return direction, scores
    end
end

# """
#     corr_track_tic(X_unit, u)

# X_unit: n×R×M unit-area, pre-CLR intensities
# u: n-vector separation score

# Returns ρ_tic::Vector{Float64} length R (corr per RI of TIC with u).
# """
function corr_track_tic(X_unit::Array{<:Real,3}, u::AbstractVector)
    n, R, M = size(X_unit)
    @assert length(u) == n
    tic = dropdims(sum(X_unit, dims=3); dims=3) # n×R
    ρ = Vector{Float64}(undef, R)
    @inbounds for r in 1:R
        ρ[r] = robustcor(@view(tic[:, r]), u)
    end
    ρ
end