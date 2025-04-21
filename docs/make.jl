using Documenter, StatisticalProjections

DocMeta.setdocmeta!(StatisticalProjections, :DocTestSetup, :(using StatisticalProjections); recursive=true)

makedocs(
	sitename = "StatisticalProjections",
	format = Documenter.HTML(),
	modules = [StatisticalProjections]
)

deploydocs(
	repo = "github.com/oniehuis/StatisticalProjections.jl"
)