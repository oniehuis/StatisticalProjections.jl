module StatisticalProjections

using LinearAlgebra
using Optim
using Random
using Statistics
using StatsBase

include("CPPLS_model.jl")

export CPPLS
export CPPLSLight
export fit_cppls
export fit_cppls_light
export predict
export one_hot_argmax
export calculate_scores
export cca_coeffs_and_corr

include("CPPLS_cv.jl")
export nested_cv_with_permutation
export nested_cv_permutation
export calculate_p_value
export nested_cv
export nmc

include("CPPLS_utilities.jl")

export clr_transform
export clr_transform!
export labels_to_one_hot

end # module StatisticalProjections
