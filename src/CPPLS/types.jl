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
