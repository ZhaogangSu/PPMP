# PPMP Solver

A Julia package for solving the Probabilistic Perfect Matching Problem (PPMP), developed and maintained by Zhaogang Su (suzhaogang@lsec.cc.ac.cn).

## Problem Definition

The Probabilistic Perfect Matching Problem (PPMP) is a variant of the classical perfect matching problem under uncertainty. Given:

- A bipartite graph G = (V₁ ∪ V₂, E) with edge costs
- A set of scenarios, where each scenario represents a subgraph of G
- Probabilities associated with each scenario
- A risk tolerance parameter ε ∈ [0,1]

The goal is to:

1. Select a subset of edges E' ⊆ E with minimum total cost
2. Such that the probability of being able to form a perfect matching using selected edges E' in a random scenario is at least 1-ε

## Requirements

- Julia 1.10 or higher
- CPLEX 20.1.0 / 22.1.0
  - Set CPLEX environment variable before using:
    ```bash
    export CPLEX_STUDIO_BINARIES="/path/to/cplex/bin/x86-64_linux/"
    ```
  - Add this to your `.bashrc` or `.zshrc` for permanent setup
- Required Julia packages listed in Project.toml (installed automatically)

## Quick Start Guide

### 1. Installation

Clone and set up the project:
```bash
git clone [repository-url]
cd PPMP
```

Start Julia REPL with the project:
```bash
julia --project=.
```

Install dependencies:
```julia
using Pkg
Pkg.instantiate()

using PPMP
```

Or later you can use
```bash
julia --project=. -ie "using Pkg; Pkg.instantiate(); using PPMP;"
```
to directly initialize.

### 2. Generate Test Instances

In Julia REPL:
```julia
using PPMP

# Generate a standard test set
```

This will create instances in `data/grid/` with various sizes and parameters. For a quick test, we'll use a small instance:
```
data/grid/test_20_16/d0.3/f0.3/test_20_16_d0.3_f0.3_se1.json
```

### 3. Basic Solving

Let's solve a specific instance with both solvers. In Julia REPL:

```julia
# Define instance path
path = "data/grid/test_20_16/d0.3/f0.3/test_20_16_d0.3_f0.3_se1.json"

# Solve with Original MIP Solver (OMS)
julia_main(["OriginalMIPSolver", "0.1", path, ".", "--max-time", "3600"])

# Solve with Flow Cut Solver (FCS)
julia_main(["FlowCutSolver", "0.1", path, ".", "--max-time", "3600"])
```

### 4. Advanced Settings

By default, FCS uses presolve and mixing inequalities, and OMS uses CPLEX's Auto-Benders decomposition. To modify these features:

```julia
# Disable presolve for FCS
julia_main(["FlowCutSolver", "0.1", path, ".",
           "--is-presolve-flowcut", "false"])

# Disable mixing inequalities for FCS
julia_main(["FlowCutSolver", "0.1", path, ".",
           "--lazy-mixing-cuts-enabled", "false",
           "--user-mixing-cuts-enabled", "false"])

# Adjust cut generation for FCS
julia_main(["FlowCutSolver", "0.1", path, ".",
           "--root-user-cuts-per-node", "4000",
           "--tree-lazy-cuts-per-node", "2"])

# Disable Auto-Benders for OMS
julia_main(["OriginalMIPSolver", "0.1", path, ".",
           "--auto-benders", "false"])

## Precompilation

To reduce startup time or create a portable executable, you can create a system image or standalone application. In Julia REPL:

```julia
# Create system image
include("scripts/precompile/create_sysimage.jl")

# Or create standalone application
include("scripts/precompile/create_app.jl")
```

After creating the system image, start Julia with:
```bash
julia --sysimage lib/PPMP.so --project=.
```

### Using Compiled Executable

After creating the standalone application, you can run the solver directly from command line:

```bash
# Using Original MIP Solver
./bin/bin/PPMP OriginalMIPSolver 0.1 path/to/instance.json results/ --max-time 3600

# Using Flow Cut Solver
./bin/bin/PPMP FlowCutSolver 0.1 path/to/instance.json results/ --max-time 3600
```

The compiled executable accepts the same arguments as the Julia REPL version, but can be run directly from the command line without starting Julia.

## Command Line Arguments

Essential arguments:
```
1. solver_type:    "OriginalMIPSolver" or "FlowCutSolver"
2. epsilon:        Risk tolerance (0-1)
3. instance_path:  Path to instance file
4. result_dir:     Directory for results
```

Common optional flags:
```
--max-time              Time limit in seconds (default: 14400)
--threads               Number of threads (default: 1)
--mip-gap               Target optimality gap (default: 1e-4)
--print-level           Output detail level (0-3, default: 0)
```

## Results

Solutions are saved in the specified result directory with the format:
```
{instance_name}_eps{epsilon}_{solver}.json
```

For example:
```
test_20_16_d0.3_f0.3_se1_eps0.1_FCS.json
```

## Documentation

For more details on:
- Instance generation parameters: See `config/testset_settings.jl`
- Solver configuration: See `config/experiment_settings.jl`
- Default settings: Use `help> PPMPConfig` in REPL

## Project Structure

```
.
├── src/                        # Main source code
│   ├── PPMP.jl                # Main module file
│   ├── algorithms/            # Core algorithms
│   │   ├── base.jl           # Base functionality
│   │   ├── original_mip.jl   # Original MIP formulation
│   │   ├── flow_cut.jl       # Flow cut algorithm
│   │   ├── mixing_ineq.jl    # Mixing inequalities
│   │   └── presolve.jl       # Presolve techniques
│   ├── callbacks/            # CPLEX callbacks
│   │   ├── flow_cut_cb.jl    # Flow cut callbacks
│   │   └── mixing_ineq_cb.jl # Mixing inequalities callbacks
│   ├── types/               # Type definitions
│   │   ├── problem_types.jl  # Problem data structures
│   │   ├── solution_types.jl # Solution data structures
│   │   ├── callback_types.jl # Callback data structures
│   │   └── presolve_types.jl # Presolve data structures
│   └── utils/               # Utility functions
│       ├── data_generation.jl  # Instance generation
│       ├── data_processing.jl  # Data handling
│       ├── result_logging.jl   # Result output
│       └── instance_runner.jl  # Main solving process
├── config/                    # Configuration files
│   ├── experiment_settings.jl # Solver parameters
│   └── testset_settings.jl   # Test instance parameters
├── scripts/
│   └── precompile/          # Precompilation scripts
├── test/                     # Test files
└── Project.toml             # Project dependencies
```

## Contact

Zhaogang Su - suzhaogang@lsec.cc.ac.cn
