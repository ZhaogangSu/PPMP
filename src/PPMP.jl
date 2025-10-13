# src/PPMP.jl

module PPMP

# Import required packages
using JuMP
import MathOptInterface as MOI
using CPLEX
using Graphs
using DataStructures
using JSON
using Statistics
using ArgParse
using JSON
using Random

# Re-exports
export MOI

# Include files
# Configuration
include("../config/experiment_settings.jl")
include("../config/testset_settings.jl")

# Type definitions
include("types/callback_types.jl")
include("types/problem_types.jl")
include("types/solution_types.jl")
include("types/presolve_types.jl")

# Core algorithms
include("algorithms/base.jl")
include("algorithms/original_mip.jl")
include("algorithms/flow_cut.jl")
include("algorithms/mixing_ineq.jl")
include("algorithms/presolve.jl")

# Callbacks
include("callbacks/flow_cut_cb.jl")
include("callbacks/mixing_ineq_cb.jl")

# Utilities
include("utils/data_generation.jl")
include("utils/data_processing.jl")
# include("utils/restart_utils.jl")
include("utils/result_logging.jl")
include("utils/instance_runner.jl")

export TESTSET_CONFIGS,
       INSTANCE_SIZES, EDGE_DENSITIES, FAILURE_PROBABILITIES,
       EXPLORATIVE_INSTANCE_SIZES, EXPLORATIVE_EDGE_DENSITIES, EXPLORATIVE_FAILURE_PROBABILITIES, 
       LARGE_INSTANCE_SIZES, LARGE_EDGE_DENSITIES, LARGE_FAILURE_PROBABILITIES


# Configuration exports (from experiment_settings.jl)
export PPMPConfig, 
       default_config, 
       config_from_args, 
       parse_commandline,
       get_depth_scaling_factor, ScalingType, EXPONENTIAL, LINEAR, STEP, HYPERBOLIC


# Core type exports (from problem_types.jl)
export PPMPInstance,
       PPMPSolver,
       CutGenerator,
       RawInstanceData

# Solution type exports (from solution_types.jl)
export CutPool,
       CallbackStats,
       StoredCut,
       FlowCutGlbPool,
       PPMPSolution,
       SolutionStats,
       Cut,
       SolutionProgress

# presolve type exports (from presolve_types.jl)
export MergeStats,
       PresolveInfo,
       PresolveStats

# Solver exports (from original_mip.jl and flow_cut.jl)
export OriginalMIPSolver,
       FlowCutSolver,
       solve

# Algorithm and utility exports (from base.jl)
export create_flow_network,
       print_model_details,
       create_flow_network_capacitated,
       create_flow_network_capacitated_costed,
       export_to_mps,
       update_network_capacities!
    

# Cut finding exports (from flow_cut.jl)
export find_violated_cuts,
       custom_mincut
       
    
# Mixing Cut finding exports (from mixing_ineq.jl)
export compute_q_kk_prime_MN,
       generate_mixing_coefficients

# Presolve exports (from presolve.jl)
export PresolveInfo,
       ppmp_presolve,
       find_dominated_scenarios,
       create_reduced_instance,
       print_presolve_results


# Callback exports (from flow_cut_cb.jl)
export  store_cut,
        flow_lazy_cons_callback,
        flow_user_cuts_callback
    #    generate_flow_cuts_callback,
    #    print_callback_stats,
    #    print_solution_stats,
    #    test_solve_with_logging,
    #    jump_flow_lazy_cons_callback,
    #    jump_flow_user_cuts_callback

# Callback exports (from mixing_ineq_cb.jl)
export process_mixing_cuts_lazy!,
       process_mixing_cuts_user!


# Data generation and processing exports (from data_generation.jl and data_processing.jl)
export generate_bipartite_with_scenarios,
       generate_instances,
       generate_uniform_bipartite_with_scenarios,
       generate_uniform_instances,
       generate_presolve_bipartite_with_scenarios,
       generate_presolve_instances,
       save_instance_to_json,
       create_test_instance,
       create_test_ppmp_instance,
       create_test_presolve_instance,
       create_test_ppmp_presolve_instance,
       print_instance_stats,
       load_instance_from_json,
       merge_identical_scenarios,
       print_scenario_merge_statistics

# Result handling exports (from result_logging.jl and instance_runner.jl)
export ResultData,
       create_result_data,
       save_results,
       run_instance,
       init_run_instance,
       update_node_limit,
       update_restart_flag,
       update_callbackstats_pool,
       merge_solution_stats,
       write_round_solve_info

export EDGE_DENSITIES, FAILURE_PROBABILITIES, INSTANCE_SIZES, RANDOM_SEEDS

"""
Main entry point for the executable
"""
function julia_main(args::Vector{String}=ARGS)::Cint
    try
        # Parse arguments
        s = parse_commandline()
        @info "Parsing command line arguments"
        parsed_args = parse_args(args, s)
        @info "Arguments parsed"
        # Create configuration
        config = config_from_args(parsed_args)
        @info "Configuration created"
        
        # Parse solver type
        solver_str = parsed_args["solver_type"]
        solver_type = if solver_str == "OriginalMIPSolver"
            OriginalMIPSolver
        elseif solver_str == "FlowCutSolver"
            FlowCutSolver
        else
            throw(ArgumentError("Solver type must be either 'OriginalMIPSolver' or 'FlowCutSolver'"))
        end

        # Extract required arguments
        epsilon = parsed_args["epsilon"]
        instance_path = parsed_args["instance_path"]
        result_dir = parsed_args["result_dir"]
        # print(parsed_args)
        
        # Validate epsilon
        if !(0 <= epsilon <= 1)
            throw(ArgumentError("Epsilon must be between 0 and 1"))
        end

        # Run instance
        run_instance(solver_type, epsilon, instance_path, result_dir, config)
        @info "Instance run complete"
        return 0  # success
    catch e
        if isa(e, ArgumentError)
            println(stderr, "Input error: ", e)
        else
            println(stderr, "Execution error: ", e)
            println(stderr, "Stacktrace:")
            println(stderr, stacktrace())
        end
        return 1  # failure
    end
end

export julia_main

end # module