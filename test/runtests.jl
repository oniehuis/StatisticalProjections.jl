using StatisticalProjections
using Test

@testset "CPPLS/types.jl" begin
    include(joinpath("CPPLS", "types.jl"))
end

@testset "CPPLS/predict" begin
    include(joinpath("CPPLS", "predict.jl"))
end

@testset "CPPLS/cca" begin
    include(joinpath("CPPLS", "cca.jl"))
end

@testset "CPPLS/fit" begin
    include(joinpath("CPPLS", "fit.jl"))
end

@testset "CPPLS/preprocessing" begin
    include(joinpath("CPPLS", "preprocessing.jl"))
end

@testset "Utils/encoding" begin
    include(joinpath("Utils", "encoding.jl"))
end

@testset "Utils/matrix" begin
    include(joinpath("Utils", "matrix.jl"))
end

@testset "Utils/statistics" begin
    include(joinpath("Utils", "statistics.jl"))
end

@testset "Evaluation/crossvalidation" begin
    include(joinpath("Evaluation", "crossvalidation.jl"))
end

@testset "Evaluation/metrics" begin
    include(joinpath("Evaluation", "metrices.jl"))
end
