using Documenter, StatisticalProjections

DocMeta.setdocmeta!(StatisticalProjections, :DocTestSetup, :(using StatisticalProjections); recursive=true)

makedocs(
	sitename = "StatisticalProjections",
	format = Documenter.HTML(),
	modules = [StatisticalProjections],
	authors = "Oliver Niehuis",
	pages = [
        "Home" => "index.md",
		"CPPLS" => Any[
			"CPPLS/fit.md",
            "CPPLS/types.md",
			"CPPLS/predict.md",
		    ],
		"Evaluation" => Any[
			"Evaluation/metrics.md",
			"Evaluation/crossvalidation.md",
		    ],
		"Utils" => Any[
			"Utils/encoding.md",
			"Utils/matrix.md",
			"Utils/statistics.md",
		    ]
    ]
)

deploydocs(
	repo = "github.com/oniehuis/StatisticalProjections.jl"
)