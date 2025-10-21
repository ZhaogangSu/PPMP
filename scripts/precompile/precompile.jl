# scripts/precompile/precompile.jl

using PPMP
using ArgParse

# We use the instance mentioned in README.md for precompile
path = "data/grid/test_20_16/d0.3/f0.3/test_20_16_d0.3_f0.3_se1.json"

julia_main(["OriginalMIPSolver", "0.05", path, ".", "--max-time", "35"])
julia_main(["FlowCutSolver", "0.1", path, ".",
           "--merge-identical-scenarios", "false",
           "--is-presolve-flowcut", "false",
           "--mixing-print-level", "2",
           "--max-time", "300"])