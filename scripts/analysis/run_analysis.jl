# scripts/analysis/run_analysis.jl

using PPMP
using CCAPDataAnalysis

"""
Run analysis on results folder with optional filtering
"""
function run_analysis(folder::String, filter_pattern::String="")
    # Get results
    if isempty(filter_pattern)
        results = collect_results(folder)
        output_dir = folder
    else
        results = collect_results(folder, filter_pattern)
        output_dir = joinpath(folder, filter_pattern)
    end
    
    # Process results
    df = results_to_dataframe(results)
    
    # Create output directories
    processed_dir = joinpath("results", "processed", basename(output_dir))
    figures_dir = joinpath("results", "figures", basename(output_dir))
    mkpath(processed_dir)
    mkpath(figures_dir)
    
    # Generate and save statistics and figures
    stats = generate_summary_stats(df, save_path=joinpath(processed_dir, "statistics.csv"))
    visualize_solving_times(df, save_path=figures_dir)
    
    return df, stats
end


eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa_2345

julia --project=. -ie "using Pkg; Pkg.instantiate(); using CCAPDataAnalysis;"
julia --project=. -ie "using Pkg; Pkg.instantiate(); using PPMP;"


# 
path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se1.json"
julia_main(["FlowCutSolver", "0.01", 
path, "../", 
"--is-presolve-flowcut", "false"
])
#
path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se1.json"
julia_main(["FlowCutSolver", "0.01", 
path, "./", 
"--is-presolve-flowcut", "false",
"--print-level", "0"])

path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se1.json"
julia_main(["FlowCutSolver", "0.01", 
       path, "./", 
       "--is-presolve-flowcut", "false",
       "--print-level", "0", "--root-max-user-mixing-cuts-submit-per-round", "1"])


path = "data/dev3/test_40_200/d0.5/f0.005/test_40_200_d0.5_f0.005_se1.json"
julia_main(["FlowCutSolver", "0.3", 
            path, "../../", 
            "--is-presolve-flowcut","true",
            "--is-presolve-fix-scenario", "false"
            ])


#
path = "data/dev3/test_40_200/d0.5/f0.005/test_40_200_d0.5_f0.005_se1.json"
julia_main(["FlowCutSolver", "0.3", 
            path, ".", 
            "--is-presolve-flowcut","true",
            ])





path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se1.json"
julia_main(["FlowCutSolver", "0.01", 
path, "../", 
"--is-presolve-flowcut", "false"
])


path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se1.json"
julia_main(["FlowCutSolver", "0.01", 
path, "./", 
"--is-presolve-flowcut", "false",
"--root-max-user-mixing-cuts-submit-per-round", "5", 
"--root-max-user-mixing-cuts-per-round", "5"
])


julia_main(["FlowCutSolver", "0.01", 
       path, "./", 
       "--is-presolve-flowcut", "false",
       "--root-max-user-mixing-cuts-submit-per-round", "2", 
       "--root-max-user-mixing-cuts-per-round", "2"
       ])


path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se3.json"
julia_main(["FlowCutSolver", "0.01", 
       path, "./", 
       "--is-presolve-flowcut", "false",
       "--root-max-user-mixing-cuts-submit-per-round", "22", 
       "--root-max-user-mixing-cuts-per-round", "22"
       ])

path = "data/dev3/test_20_200/d0.5/f0.01/test_20_200_d0.5_f0.01_se3.json"
julia_main(["FlowCutSolver", "0.01", 
path, "../", 
"--is-presolve-flowcut", "false",
"--user-mixing-cuts-enabled", "false",
"--user-scenario-relaxation-value-threshold", "1e-6"
])


