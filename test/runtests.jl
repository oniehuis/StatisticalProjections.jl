using StatisticalProjections
using Test

@testset "CPPLS/types.jl" begin
    include(joinpath("CPPLS", "types.jl"))
end

@testset "CPPLS/predict" begin
    include(joinpath("CPPLS", "predict.jl"))
end

@testset "CPPLS/fit" begin
    include(joinpath("CPPLS", "fit.jl"))
end

@testset "Utils/encoding" begin
    include(joinpath("Utils", "encoding.jl"))
end
