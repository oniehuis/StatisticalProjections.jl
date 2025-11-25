module StatisticalProjections

using LinearAlgebra
using Optim
using Random
using Statistics
using StatsBase

include("CPPLS/types.jl")
include("CPPLS/preprocessing.jl")
include("CPPLS/cca.jl")
include("CPPLS/fit.jl")
include("CPPLS/predict.jl")
include("CPPLS/metrics.jl")
include("CPPLS/crossvalidation.jl")

include("Utils/encoding.jl")
include("Utils/matrix.jl")
include("Utils/statistics.jl")

export CPPLS
export CPPLSLight
export fit_cppls
export fit_cppls_light
export predict
export predictonehot
export project
export nested_cv_permutation
export nested_cv
export calculate_p_value
export separationaxis
export fisherztrack
export labels_to_one_hot
export one_hot_to_labels
export find_invariant_and_variant_columns

end # module StatisticalProjections
