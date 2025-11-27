module MakieExtension

using Makie
using StatisticalProjections

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "ext", "makie_extensions", "cppls.jl"))

end
