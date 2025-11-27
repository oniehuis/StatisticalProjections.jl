using Documenter
using Markdown
using Makie
using StatisticalProjections

function ensure_makie_extension!()
    ext = Base.get_extension(StatisticalProjections, :MakieExtension)
    return ext !== nothing ? ext :
           Base.require_extension(StatisticalProjections, :MakieExtension)
end

ensure_makie_extension!()

DocMeta.setdocmeta!(
    StatisticalProjections,
    :DocTestSetup,
    :(using StatisticalProjections);
    recursive = true,
)

makedocs(
    sitename = "StatisticalProjections",
    format = Documenter.HTML(),
    modules = [StatisticalProjections],
    authors = "Oliver Niehuis",
    pages = [
        "Home" => "index.md",
        "CPPLS" => Any[
            "CPPLS/types.md",
            "CPPLS/fit.md",
            "CPPLS/predict.md",
            "CPPLS/crossvalidation.md",
            "CPPLS/visualization.md",
            "CPPLS/internal.md",
        ],
        "Utils" => Any[
            "Utils/encoding.md",
            "Utils/matrix.md",
            "Utils/statistics.md",
            "Utils/internal.md",
        ],
    ],
)

deploydocs(repo = "github.com/oniehuis/StatisticalProjections.jl", devbranch = "main")
