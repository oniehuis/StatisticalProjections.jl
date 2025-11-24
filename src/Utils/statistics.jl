@inline function robustcor(x::AbstractVector, y::AbstractVector)
    (std(x) == 0 || std(y) == 0) && return 0.0
    c = cor(x, y)
    isfinite(c) ? c : 0.0
end


function fisherztrack(X::AbstractArray{<:Real,3},
                      scores::AbstractVector{<:Real};
                      weights::Symbol=:mean)

    n_samples, n_axis₁, n_axis₂ = size(X)
    length(scores) == n_samples ||
        throw(ArgumentError("scores length must equal size(X, 1)."))
    (weights === :mean || weights === :none) ||
        throw(ArgumentError("weights must be :mean or :none."))

    ρ = Vector{Float64}(undef, n_axis₁)

    lo = nextfloat(-1.0)
    hi = prevfloat(1.0)

    @inbounds for a₁ in 1:n_axis₁
        rs = Vector{Float64}(undef, n_axis₂)
        ws = ones(Float64, n_axis₂)

        for a₂ in 1:n_axis₂
            xs = @view X[:, a₁, a₂]
            rs[a₂] = robustcor(xs, scores)
            if weights === :mean
                ws[a₂] = mean(xs)
            end
        end

        zs = atanh.(clamp.(rs, lo, hi))
        z̄ = sum(ws .* zs) / (sum(ws) + eps(Float64))
        ρ[a₁] = tanh(z̄)
    end

    ρ
end


function separationaxis(Xscores::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
                        method::Symbol=:centroid, positive_class::T=1) where {T<:Integer}

    n, p = size(Xscores)
    p ≥ 1 || throw(ArgumentError("Expecting at least one score axis (got $p)."))
    size(Y, 1) == n || throw(ArgumentError("Row count mismatch between Xscores and Y."))
    size(Y, 2) == 2 || throw(ArgumentError("Binary only; for K > 2 use one-vs-rest."))

    all((Y .== 0) .| (Y .== 1)) ||
        throw(ArgumentError("Y must be strictly one-hot (0/1)."))
    all(sum(Y, dims=2) .== 1) ||
        throw(ArgumentError("Each row of Y must have exactly one '1'."))

    idx₁ = findall(Y[:, 1] .== 1)
    idx₂ = findall(Y[:, 2] .== 1)
    (!isempty(idx₁) && !isempty(idx₂)) ||
        throw(ArgumentError("Both classes must be present."))

    if p == 1
        s = vec(@view Xscores[:, 1])
        direction = ones(eltype(Xscores), 1)
        scores = copy(s)

        m₁, m₂ = mean(scores[idx₁]), mean(scores[idx₂])
        want   = positive_class == 1 ? m₁ : m₂
        other  = positive_class == 1 ? m₂ : m₁
        if want < other
            direction .= -direction
            scores    .= -scores
        end
        
        return direction, scores

    else
        X₁ = @view Xscores[idx₁, :]
        X₂ = @view Xscores[idx₂, :]
        μ₁ = vec(mean(X₁, dims=1))
        μ₂ = vec(mean(X₂, dims=1))

        direction = if method === :centroid
            μ₁ - μ₂
        elseif method === :lda
            S₁ = cov(X₁; corrected=true)
            S₂ = cov(X₂; corrected=true)
            (S₁ + S₂) \ (μ₁ - μ₂)
        else
            throw(ArgumentError("method must be :centroid or :lda"))
        end

        if !any(isfinite, direction) || norm(direction) == 0
            throw(ArgumentError("Separation axis is undefined (zero vector). Check class means."))
        end

        direction ./= (norm(direction) + eps(eltype(direction)))
        scores = Xscores * direction

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


function corr_track_tic(X_unit::Array{<:Real,3}, u::AbstractVector)
    n, R, M = size(X_unit)
    @assert length(u) == n
    tic = dropdims(sum(X_unit, dims=3); dims=3)
    ρ = Vector{Float64}(undef, R)
    @inbounds for r in 1:R
        ρ[r] = robustcor(@view(tic[:, r]), u)
    end
    ρ
end
