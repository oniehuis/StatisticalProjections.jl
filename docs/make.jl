using Documenter, StatisticalProjections

DocMeta.setdocmeta!(StatisticalProjections, :DocTestSetup, :(using StatisticalProjections); recursive=true)

makedocs(
	sitename = "StatisticalProjections",
	format = Documenter.HTML(),
	modules = [StatisticalProjections],
	authors = "Oliver Niehuis",
	pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
	repo = "github.com/oniehuis/StatisticalProjections.jl"
)