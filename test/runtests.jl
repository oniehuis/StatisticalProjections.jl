using StatisticalProjections
using Test

@testset "CPPLS/types.jl" begin
    include(joinpath("CPPLS", "types.jl"))
end

@testset "CPPLS/predict" begin
    include(joinpath("CPPLS", "predict.jl"))
end
