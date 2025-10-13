# config/experiment_settings.jl
# Experiment configuration parameters

using ArgParse

"""
Enum for different types of scaling functions
"""
@enum ScalingType begin
    EXPONENTIAL = 1
    LINEAR = 2
    STEP = 3
    HYPERBOLIC = 4  # 1 / (1 + decay_rate * depth)
end


"""
Calculate cut limit scaling factor based on depth and scaling type
"""
function get_depth_scaling_factor(depth::Int, scaling_type::ScalingType, 
                                decay_rate::Float64, min_factor::Float64) # decay_rate = 0.3 means no user cuts >= depth 18
    if scaling_type == EXPONENTIAL
        return max(min_factor, exp(-decay_rate * depth))
    elseif scaling_type == LINEAR
        return max(min_factor, 1.0 - decay_rate * depth)
    elseif scaling_type == STEP
        if depth < 5
            return 1.0
        elseif depth < 10
            return 0.5
        else
            return min_factor
        end
    elseif scaling_type == HYPERBOLIC
        return max(min_factor, 1.0 / (1.0 + decay_rate * depth))
    end
    # Add a default return to ensure function always returns a value
    return 1.0
end


"""
Main configuration structure for PPMP
"""
struct PPMPConfig
    # System configuration
    threads::Int
    max_time::Float64
    mip_gap::Float64
    mip_gap_init::Float64
    max_node::Int
    max_memory_mb::Int
    
    # Basic solver configuration
    create_from_scratch::Bool  # create solver from scratch: No initial cuts
    create_left_cons::Bool     # if not create from scratch: create left cons
    create_right_cons::Bool    # if not create from scratch: create right cons
    create_random_cons::Bool   # if not create from scratch: create random cons

    presolve_enabled::Bool
    cut_generation::String  # "none", "static", "dynamic"
    separation_method::String  # "exact", "heuristic"
    callback_print_level::Int
    mixing_print_level::Int # print level for mixing cuts

    # CPLEX solver parameters
    cpx_presolve::Int   
    auto_benders::Bool  # auto benders decomposition for original solver
    
    # user cuts / lazy cons opening 
    user_callback_open::Bool  # open user cuts

    # Lazy constraint parameters
    lazy_max_scenarios::Int          # Maximum scenarios to check for lazy cuts
    root_lazy_cuts_per_node::Int     # Maximum lazy cuts per callback (for computational efficiency)
    root_lazy_scenario_whole_search::Bool  # whether use whole search for root lazy
    tree_lazy_cuts_per_node::Int     # Maximum lazy cuts per callback (for computational efficiency)
    tree_lazy_scenario_whole_search::Bool  # whether use whole search for tree lazy

    lazy_min_violation::Float64      # Minimum violation for lazy cuts
    
    # Root node user cuts
    root_user_max_cuts_per_node::Int     # Maximum user cuts at root
    root_user_max_cuts_per_round::Int # Maximum user cuts per callback at root
    root_user_max_rounds_per_node::Int   # Maximum user callback round per at root
    root_user_scenario_whole_search::Bool  # whether use whole search for root user
    root_user_max_scenarios::Int         # Maximum scenarios to check for user cuts at root
    root_user_min_violation::Float64     # Minimum violation for user cuts at root
    user_scenario_relaxation_value_threshold::Float64  # Minimum scenario relaxation value to attemp to find violated user cuts for this scenario
    
    # Tree node user cuts
    tree_user_max_cuts_per_node::Int     # Maximum user cuts per node in tree
    tree_user_max_cuts_per_round::Int # Maximum user cuts per callback in tree
    tree_user_max_rounds_per_node::Int   # Maximum user callback round per in tree
    tree_user_scenario_whole_search::Bool  # whether use whole search for tree user
    tree_user_max_scenarios::Int         # Maximum scenarios to check for user cuts in tree
    tree_user_min_violation::Float64     # Minimum violation for user cuts in tree
    tree_user_frequency::Int             # Generate user cuts every N nodes

    # Mixing cuts parameters
    lazy_mixing_cuts_enabled::Bool           # Whether to use lazy_mixing inequalities
    lazy_mixing_cuts_violation_threshold::Float64  # Minimum violation for submitting lazy mixing cuts
    root_max_lazy_mixing_cuts_per_round::Int      # Maximum number of lazy mixing cuts to compute at root node per callback
    root_max_lazy_mixing_cuts_submit_per_round::Int      # Maximum number of lazy mixing cuts to submit at root node per callback
    tree_max_lazy_mixing_cuts_per_round::Int      # Maximum number of lazy mixing cuts to compute at tree node per callback
    tree_max_lazy_mixing_cuts_submit_per_round::Int      # Maximum number of lazy mixing cuts to submit at tree node per callback


    user_cuts_enabled::Bool                  # Whether to use user cuts
    user_mixing_cuts_enabled::Bool           # Whether to use lazy_mixing inequalities
    user_mixing_cuts_violation_threshold::Float64  # Minimum violation for submitting user mixing cuts
    user_mixing_cuts_normed_violation_rel_threshold::Float64  # Minimum relative normed_violation for submitting user mixing cuts (relative to the user cut's normed violation)
    root_max_user_mixing_cuts_per_round::Int      # Maximum number of user mixing cuts to compute at root node per callback
    root_max_user_mixing_cuts_submit_per_round::Int      # Maximum number of user mixing cuts to submit at root node per callback
    tree_max_user_mixing_cuts_per_round::Int      # Maximum number of user mixing cuts to compute at tree node per callback
    tree_max_user_mixing_cuts_submit_per_round::Int      # Maximum number of user mixing cuts to submit at tree node per callback

    max_scenarios_mixing::Int           # Maximum number of scenarios to consider for mixing
 
    # Cut pool management
    max_total_cuts::Int                  # Maximum total cuts allowed
    max_total_user_cuts::Int             # Maximum total user cuts allowed
    max_total_lazy_cuts::Int             # Maximum total lazy cuts allowed
    cut_pool_size::Int         # Maximum cuts to keep in pool
    cut_cleanup_frequency::Int  # Clean up cut pool every N nodes

    # Depth scaling parameters
    scaling_type::ScalingType
    depth_decay_rate::Float64
    min_scaling_factor::Float64
    
    # Heuristic settings
    heuristic_enabled::Bool
    heuristic_frequency::Int   # Run heuristic every N nodes
    heuristic_max_difference::Float64  # Maximum difference for rounding

    # Restart settings
    gap_multi_factor::Float64  # Multiplier for gap for each restart
    node_limit_factor::Float64  # Multiplier for node limit for each restart
    is_root_restart::Bool       # Restart at root node
    root_restart_rounds::Int         # Number of rounds to restart at root node
    is_tree_restart::Bool       # Restart at tree node
    tree_restart_rounds::Int    # Number of rounds to restart at tree node
    max_node_to_tree_restart::Int # Number of nodes to restart at tree node
    lazy_select_ratio::Float64  # Ratio of lazy cuts to select from global lazy pool
    user_select_ratio::Float64  # Ratio of user cuts to select from global user pool
    lazy_pool_growth_tolerance::Float64  # Tolerance for lazy pool growth

    # Presolve settings
    is_presolve_flowcut::Bool     # Whether to use presolve
    is_presolve_fix_scenario::Bool     # Whether to fix scenario variables in modeling based on presolve info
    is_presolve_original::Bool     # Whether to use presolve

    # logging settings
    is_logging_round_info::Bool  # Whether to log round info
end

"""
Create default configuration
"""
function default_config()
    PPMPConfig(
        # System
        1,        # threads
        3600,    # max_time (4 hours)
        0.0,     # mip_gap
        0.1,     # mip_gap_init
        10000000,      # max_node
        4096,     # max_memory_mb
        
        # Basic solver
        false,     # create_from_scratch
        true,      # create_left_cons
        true,      # create_right_cons
        false,      # create_random_cons

        true,     # presolve_enabled
        "dynamic", # cut_generation
        "exact",   # separation_method
        0,        # callback_print_level
        0,        # mixing_print_level

        # CPLEX Solver parameters
        1,          # cpx_presolve
        true,      # auto_benders

        # user cuts / lazy cons opening
        true,      # user_callback_open
        
        # lazy constraints parameters
        typemax(Int),      # lazy_max_scenarios
        typemax(Int),      # root_lazy_cuts_per_node
        true,              # root_lazy_scenario_whole_search
        1,                 # tree_lazy_cuts_per_node
        true,              # tree_lazy_scenario_whole_search
        1e-12,             # lazy_min_violation

        # Root user cuts
        2000,     # root_user_max_cuts_per_node
        100,      # root_user_max_cuts_per_round
        50,       # root_user_max_rounds_per_node
        true,     # root_user_scenario_whole_search
        1000,      # root_user_max_scenarios
        1e-1,     # root_user_min_violation
        1e-2,     # user_scenario_relaxation_value_threshold
        
        # Tree user cuts
        0,        # tree_user_max_cuts_per_node # so far we shut down the tree user cuts
        1,         # tree_user_max_cuts_per_round
        2,          # tree_user_max_rounds_per_node
        true,     # tree_user_scenario_whole_search
        5,        # tree_user_max_scenarios
        1e-1,       # tree_user_min_violation
        2,          # tree_user_frequency

        # Mixing cuts
        false,   # lazy_mixing_cuts_enabled
        1e-3,    # lazy_mixing_cuts_violation_threshold
        1,      # root_max_lazy_mixing_cuts_per_round
        1,      # root_max_lazy_mixing_cuts_submit_per_round
        0,      # tree_max_lazy_mixing_cuts_per_round
        0,      # tree_max_lazy_mixing_cuts_submit_per_round
         
        true,   # user_cuts_enabled
        true,   # user_mixing_cuts_enabled
        1e-3,    # user_mixing_cuts_violation_threshold
        1e-1,    # user_mixing_cuts_normed_violation_rel_threshold
        5,      # root_max_user_mixing_cuts_per_round
        1,      # root_max_user_mixing_cuts_submit_per_round
        0,      # tree_max_user_mixing_cuts_per_round
        0,      # tree_max_user_mixing_cuts_submit_per_round

        20,     # max_scenarios_mixing
        
        # Cut pool - maximized for lazy cuts
        typemax(Int),  # max_total_cuts
        100000,         # max_total_user_cuts (unchanged)
        typemax(Int),  # max_total_lazy_cuts
        typemax(Int),  # cut_pool_size
        100,           # cut_cleanup_frequency # not used

        # Depth scaling
        EXPONENTIAL,  # scaling_type
        0.04,          # depth_decay_rate
        1e-5,          # min_scaling_factor
        
        # Heuristic
        true,     # heuristic_enabled
        10,       # heuristic_frequency
        0.3,       # heuristic_max_difference

        # restart
        0.1,      # gap_multi_factor
        4,      # node_limit_factor
        false,      # is_root_restart
        1,         # root_restart_rounds
        false,     # is_tree_restart
        1,         # tree_restart_rounds
        500,        # max_node_to_tree_restart (0 means no tree node restart)



    
        1.0,       # lazy_select_ratio
        0.1,        # user_select_ratio
        0.01,       # lazy_pool_growth_tolerance

        # Presolve settings
        true,      # is_presolve_flowcut
        true,      # is_presolve_fix_scenario
        false,      # is_presolve_original

        # logging settings
        false      # is_logging_round_info

    )
end

"""
Command line argument parsing
"""
function parse_commandline()
    s = ArgParseSettings(
        description="PPMP Solver Command Line Interface",
        version="1.0",
        add_version=true
    )

    @add_arg_table! s begin
        "solver_type"
            help = "OriginalMIPSolver or FlowCutSolver"
            required = true
            arg_type = String
        "epsilon"
            help = "Risk tolerance (0-1)"
            arg_type = Float64
            required = true
            range_tester = x -> 0 <= x <= 1
        "instance_path"
            help = "Path to instance file"
            required = true
        "result_dir"
            help = "Directory for results"
            required = true
        "--threads"
            help = "Number of threads"
            arg_type = Int
            default = default_config().threads
        "--max-time"
            help = "Time limit (seconds)"
            arg_type = Float64
            default = default_config().max_time
        "--mip-gap"
            help = "Relative MIP gap"
            arg_type = Float64
            default = default_config().mip_gap
        "--mip-gap-init"
            help = "Initial MIP gap"
            arg_type = Float64
            default = default_config().mip_gap_init
        "--max-node"
            help = "Node limit"
            arg_type = Int
            default = default_config().max_node
        "--max-memory"
            help = "Maximum memory (MB)"
            arg_type = Int
            default = default_config().max_memory_mb
        
        # Basic solver settings
        "--create-from-scratch"
            help = "Create solver from scratch"
            arg_type = Bool
            default = default_config().create_from_scratch
        "--create-left-cons"
            help = "Create left cons"
            arg_type = Bool
            default = default_config().create_left_cons
        "--create-right-cons"
            help = "Create right cons"
            arg_type = Bool
            default = default_config().create_right_cons
        "--create-random-cons"
            help = "Create random cons"
            arg_type = Bool
            default = default_config().create_random_cons
        
        # CPLEX solver parameters
        "--cpx-presolve"
            help = "CPLEX Presolve"
            arg_type = Int
            default = default_config().cpx_presolve
        "--auto-benders"
            help = "Auto Benders decomposition"
            arg_type = Bool
            default = default_config().auto_benders

        # user cuts / lazy cons opening
        "--user-callback-open"
            help = "Open user cuts"
            arg_type = Bool
            default = default_config().user_callback_open

        # Lazy constraint settings
        "--lazy-max-scenarios"
            help = "Max scenarios to check for lazy cuts"
            arg_type = Int
            default = default_config().lazy_max_scenarios
        "--root-lazy-cuts-per-node"
            help = "Max cuts to check for lazy cuts"
            arg_type = Int
            default = default_config().root_lazy_cuts_per_node
        "--root-lazy-whole-search"
            help = "Max cuts to check for lazy cuts"
            arg_type = Bool
            default = default_config().root_lazy_scenario_whole_search
        "--tree-lazy-cuts-per-node"
            help = "Max cuts to check for lazy cuts"
            arg_type = Int
            default = default_config().tree_lazy_cuts_per_node
        "--tree-lazy-whole-search"
            help = "Max cuts to check for lazy cuts"
            arg_type = Bool
            default = default_config().tree_lazy_scenario_whole_search
        "--lazy-min-violation"
            help = "Min violation for lazy cuts"
            arg_type = Float64
            default = default_config().lazy_min_violation
        
        # Root user cut settings
        "--root-user-cuts-per-node"
            help = "Max user cuts per node at root"
            arg_type = Int
            default = default_config().root_user_max_cuts_per_node
        "--root-user-cuts-per-round"
            help = "Max user cuts per callback at root"
            arg_type = Int
            default = default_config().root_user_max_cuts_per_round
        "--root-user-rounds-per-node"
            help = "Max user cut rounds per node at root"
            arg_type = Int
            default = default_config().root_user_max_rounds_per_node
        "--root-user-scenario-whole-search"
            help = "whole scenario search to check for user cuts at root"
            arg_type = Bool
            default = default_config().root_user_scenario_whole_search
        "--root-user-scenarios"
            help = "Max scenarios to check for user cuts at root"
            arg_type = Int
            default = default_config().root_user_max_scenarios
        "--root-user-violation"
            help = "Min violation for root user cuts"
            arg_type = Float64
            default = default_config().root_user_min_violation
        "--user-scenario-relaxation-value-threshold"
            help = "Minimum scenario relaxation value to attemp to find violated user cuts for this scenario"
            arg_type = Float64
            default = default_config().user_scenario_relaxation_value_threshold
            
        # Tree user cut settings
        "--tree-user-cuts-per-node"
            help = "Max user cuts per node in tree"
            arg_type = Int
            default = default_config().tree_user_max_cuts_per_node
        "--tree-user-cuts-per-round"
            help = "Max user cuts per callback in tree"
            arg_type = Int
            default = default_config().tree_user_max_cuts_per_round
        "--tree-user-rounds-per-node"
            help = "Max user cut rounds per node in tree"
            arg_type = Int
            default = default_config().tree_user_max_rounds_per_node
        "--tree-user-scenario-whole-search"
            help = "whole scenario search to check for user cuts in tree"
            arg_type = Bool
            default = default_config().tree_user_scenario_whole_search
        "--tree-user-scenarios"
            help = "Max scenarios to check for user cuts in tree"
            arg_type = Int
            default = default_config().tree_user_max_scenarios
        "--tree-user-violation"
            help = "Min violation for tree user cuts"
            arg_type = Float64
            default = default_config().tree_user_min_violation
        "--tree-cut-freq"
            help = "User cut generation frequency in tree"
            arg_type = Int
            default = default_config().tree_user_frequency
        
        # Mixing cuts settings
        "--lazy-mixing-cuts-enabled"
            help = "Enable mixing cuts"
            arg_type = Bool
            default = default_config().lazy_mixing_cuts_enabled
        "--lazy-mixing-cuts-violation-threshold"
            help = "Minimum violation for mixing cuts"
            arg_type = Float64
            default = default_config().lazy_mixing_cuts_violation_threshold
        "--root-max-lazy-mixing-cuts-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().root_max_lazy_mixing_cuts_per_round
        "--root-max-lazy-mixing-cuts-submit-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().root_max_lazy_mixing_cuts_submit_per_round
        "--tree-max-lazy-mixing-cuts-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().tree_max_lazy_mixing_cuts_per_round
        "--tree-max-lazy-mixing-cuts-submit-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().tree_max_lazy_mixing_cuts_submit_per_round
        
        "--user-cuts-enabled"
            help = "Open user cuts"
            arg_type = Bool
            default = default_config().user_cuts_enabled
        "--user-mixing-cuts-enabled"
            help = "Enable mixing cuts"
            arg_type = Bool
            default = default_config().user_mixing_cuts_enabled
        "--user-mixing-cuts-violation-threshold"
            help = "Minimum violation for mixing cuts"
            arg_type = Float64
            default = default_config().user_mixing_cuts_violation_threshold
        "--user-mixing-cuts-normed-violation-rel-threshold"
            help = "Minimum relative normed violation for submitting user mixing cuts"
            arg_type = Float64
            default = default_config().user_mixing_cuts_normed_violation_rel_threshold
        "--root-max-user-mixing-cuts-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().root_max_user_mixing_cuts_per_round
        "--root-max-user-mixing-cuts-submit-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().root_max_user_mixing_cuts_submit_per_round
        "--tree-max-user-mixing-cuts-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().tree_max_user_mixing_cuts_per_round
        "--tree-max-user-mixing-cuts-submit-per-round"
            help = "Maximum mixing cuts per round"
            arg_type = Int
            default = default_config().tree_max_user_mixing_cuts_submit_per_round
        
        "--max-scenarios-mixing"
            help = "Maximum scenarios to consider for mixing cuts"
            arg_type = Int
            default = default_config().max_scenarios_mixing
            
        # Cut pool settings
        "--max-total-cuts"
            help = "Maximum total cuts allowed"
            arg_type = Int
            default = default_config().max_total_cuts
        "--max-total-user-cuts"
            help = "Maximum total user cuts allowed"
            arg_type = Int
            default = default_config().max_total_user_cuts
        "--max-total-lazy-cuts"
            help = "Maximum total lazy cuts allowed"
            arg_type = Int
            default = default_config().max_total_lazy_cuts
        "--cut-pool-size"
            help = "Maximum cut pool size"
            arg_type = Int
            default = default_config().cut_pool_size
        "--cut-cleanup-freq"
            help = "Cut pool cleanup frequency"
            arg_type = Int
            default = default_config().cut_cleanup_frequency
        
        # Depth scaling settings
        "--scaling-type"
            help = "Scaling type (1-4)"
            arg_type = ScalingType
            default = default_config().scaling_type
        "--depth-decay-rate"
            help = "Depth decay rate"
            arg_type = Float64
            default = default_config().depth_decay_rate
        "--min-scaling-factor"
            help = "Minimum scaling factor"
            arg_type = Float64
            default = default_config().min_scaling_factor

        # Restart settings
        "--gap-multi-factor"
            help = "Gap multiplier for restarts"
            arg_type = Float64
            default = default_config().gap_multi_factor
        "--node-limit-factor"
            help = "Node limit multiplier for restarts"
            arg_type = Float64
            default = default_config().node_limit_factor
        "--is-root-restart"
            help = "Restart only at root node"
            arg_type = Bool
            default = default_config().is_root_restart
        "--root-restart-rounds"
            help = "Restart rounds"
            arg_type = Int
            default = default_config().root_restart_rounds
        "--is-tree-restart"
            help = "Restart only at root node"
            arg_type = Bool
            default = default_config().is_tree_restart
        "--max-node-to-tree-restart"
            help = "tree node restart limit"
            arg_type = Int
            default = default_config().max_node_to_tree_restart
        "--tree-restart-rounds"
            help = "Restart rounds"
            arg_type = Int
            default = default_config().tree_restart_rounds
        "--lazy-select-ratio"
            help = "Ratio of lazy cuts to select from global pool"
            arg_type = Float64
            default = default_config().lazy_select_ratio
        "--user-select-ratio"
            help = "Ratio of user cuts to select from global pool"
            arg_type = Float64
            default = default_config().user_select_ratio
        "--lazy-pool-growth-tolerance"
            help = "Tolerance for lazy pool growth"
            arg_type = Float64
            default = default_config().lazy_pool_growth_tolerance
            range_tester = x -> 0 <= x <= 1

        # Presolve settings
        "--is-presolve-flowcut"
            help = "Use flowcut presolve"
            arg_type = Bool
            default = default_config().is_presolve_flowcut
        "--is-presolve-fix-scenario"
            help = "Fix scenario variables based on presolve"
            arg_type = Bool
            default = default_config().is_presolve_fix_scenario
        "--is-presolve-original"
            help = "Use original presolve"
            arg_type = Bool
            default = default_config().is_presolve_original
        
        # logging settings
        "--is-logging-round-info"
            help = "Log round info"
            arg_type = Bool
            default = default_config().is_logging_round_info
        
        
        # Debug settings
        "--print-level"
            help = "Callback print level (0-2)"
            arg_type = Int
            default = default_config().callback_print_level
            range_tester = x -> 0 <= x <= 3
        "--mixing-print-level"
            help = "Mixing cut print level (0-2)"
            arg_type = Int
            default = default_config().mixing_print_level
            range_tester = x -> 0 <= x <= 3
    end

    return s
end


"""
Create configuration from command line arguments
"""
function config_from_args(parsed_args::Dict{String, Any}, base_config::PPMPConfig=default_config())
    PPMPConfig(
        # System configuration
        parsed_args["threads"],
        parsed_args["max-time"],
        parsed_args["mip-gap"],
        parsed_args["mip-gap-init"],
        parsed_args["max-node"],
        parsed_args["max-memory"],
        
        # Basic solver configuration
        parsed_args["create-from-scratch"],
        parsed_args["create-left-cons"],
        parsed_args["create-right-cons"],
        parsed_args["create-random-cons"],

        base_config.presolve_enabled,
        base_config.cut_generation,
        base_config.separation_method,
        parsed_args["print-level"],
        parsed_args["mixing-print-level"],
        
        # CPLEX Solver parameters
        parsed_args["cpx-presolve"],
        parsed_args["auto-benders"],
        
        # user cuts / lazy cons opening
        parsed_args["user-callback-open"],

        # lazy constraints
        parsed_args["lazy-max-scenarios"],
        parsed_args["root-lazy-cuts-per-node"],
        parsed_args["root-lazy-whole-search"],
        parsed_args["tree-lazy-cuts-per-node"],
        parsed_args["tree-lazy-whole-search"],
        parsed_args["lazy-min-violation"],
        
        # Root node user cuts
        parsed_args["root-user-cuts-per-node"],
        parsed_args["root-user-cuts-per-round"],
        parsed_args["root-user-rounds-per-node"],
        parsed_args["root-user-scenario-whole-search"],
        parsed_args["root-user-scenarios"],
        parsed_args["root-user-violation"],
        parsed_args["user-scenario-relaxation-value-threshold"],
        
        # Tree node user cuts
        parsed_args["tree-user-cuts-per-node"],
        parsed_args["tree-user-cuts-per-round"],
        parsed_args["tree-user-rounds-per-node"],
        parsed_args["tree-user-scenario-whole-search"],
        parsed_args["tree-user-scenarios"],
        parsed_args["tree-user-violation"],
        parsed_args["tree-cut-freq"],

        # Mixing cuts
        parsed_args["lazy-mixing-cuts-enabled"],
        parsed_args["lazy-mixing-cuts-violation-threshold"],
        parsed_args["root-max-lazy-mixing-cuts-per-round"],
        parsed_args["root-max-lazy-mixing-cuts-submit-per-round"],
        parsed_args["tree-max-lazy-mixing-cuts-per-round"],
        parsed_args["tree-max-lazy-mixing-cuts-submit-per-round"],

        parsed_args["user-cuts-enabled"],
        parsed_args["user-mixing-cuts-enabled"],
        parsed_args["user-mixing-cuts-violation-threshold"],
        parsed_args["user-mixing-cuts-normed-violation-rel-threshold"],
        parsed_args["root-max-user-mixing-cuts-per-round"],
        parsed_args["root-max-user-mixing-cuts-submit-per-round"],
        parsed_args["tree-max-user-mixing-cuts-per-round"],
        parsed_args["tree-max-user-mixing-cuts-submit-per-round"],

        parsed_args["max-scenarios-mixing"],
        
        # Cut pool management
        parsed_args["max-total-cuts"],
        parsed_args["max-total-user-cuts"],
        parsed_args["max-total-lazy-cuts"],
        parsed_args["cut-pool-size"],
        parsed_args["cut-cleanup-freq"],

        # Depth Scaling
        parsed_args["scaling-type"],
        parsed_args["depth-decay-rate"],
        parsed_args["min-scaling-factor"],

        # Heuristic settings
        base_config.heuristic_enabled,
        base_config.heuristic_frequency,
        base_config.heuristic_max_difference,

        # Restart settings
        parsed_args["gap-multi-factor"],
        parsed_args["node-limit-factor"],
        parsed_args["is-root-restart"],
        parsed_args["root-restart-rounds"],
        parsed_args["is-tree-restart"],
        parsed_args["tree-restart-rounds"],
        parsed_args["max-node-to-tree-restart"],
        parsed_args["lazy-select-ratio"],
        parsed_args["user-select-ratio"],
        parsed_args["lazy-pool-growth-tolerance"],

        # Presolve settings
        parsed_args["is-presolve-flowcut"],
        parsed_args["is-presolve-fix-scenario"],
        parsed_args["is-presolve-original"],

        # logging settings
        parsed_args["is-logging-round-info"]

    )
end