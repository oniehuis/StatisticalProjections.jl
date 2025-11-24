module StatisticalProjections

using LinearAlgebra
using Optim
using Random
using Statistics
using StatsBase

include("Utils/encoding.jl")
include("Utils/matrix.jl")
include("Utils/statistics.jl")

include("CPPLS/types.jl")
include("CPPLS/preprocessing.jl")
include("CPPLS/cca.jl")
include("CPPLS/fit.jl")
include("CPPLS/predict.jl")

include("Evaluation/matrics.jl")
include("Evaluation/crossvalidation.jl")

export CPPLS
export CPPLSLight
export fit_cppls
export fit_cppls_light
export predict
export predictonehot
export project
export cca_coeffs_and_corr

export nested_cv_permutation
export calculate_p_value
export nested_cv
export nmc

export labels_to_one_hot
export one_hot_to_labels
export find_invariant_and_variant_columns
export separationaxis
export fisherztrack

end # module StatisticalProjections
