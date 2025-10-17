# src/utils/instance_runner.jl


using JuMP
using JSON
import MathOptInterface as MOI
using CPLEX


# Function to calculate constraint activation level at a solution
function calculate_activation(coefficients, solution)
    # For a constraint of form a'x ≤ b or a'x ≥ b
    # activation = |a'x - b| / (1 + |b|)
    # Smaller value means more active
    if isempty(solution)
        # @info "No solution provided, returning 1.0"
        return 1.0
    end


    # @info "Calculating activation for coefficients: $coefficients, solution: $solution"
    lhs = sum(coefficients .* solution)
    # @info "finished lhs"
    rhs = 0.0
    active = abs(lhs - rhs)
    # if active > 1e-9
    #     @info "Activation: $active"
    #     @info "Amazing!!!"
    # end
    return active
end

function get_filter_lazy_cons_user_cuts(glb_pool::FlowCutGlbPool,
                                        config::PPMPConfig,
                                        previous_lazy_pool_size::Int,
                                        current_round::Int,
                                        is_final::Bool)


    @info "Filtering lazy cons and user cuts for next round..."

    max_pool_size = 20000

    # relaxation_sol = glb_pool.best_fraction_sol
    relaxation_sol = glb_pool.root_fraction_sol
    
    # Initialize variables
    filtered_lazy_cons = Vector{StoredCut}()
    filtered_user_cuts = Vector{StoredCut}()
    # @info "Relaxation solution: $relaxation_sol"
    # Process lazy constraints

    function trim_pool!(pool::CutPool, cuts_to_keep::Vector{StoredCut})
        # Clear existing data structures
        empty!(pool.fingerprints)
        empty!(pool.fingerprint_to_index)
        
        # Rebuild with only the kept cuts
        pool.cuts = cuts_to_keep
        pool.pool_size = length(cuts_to_keep)
        
        # Rebuild fingerprints and indices
        for (i, cut) in enumerate(cuts_to_keep)
            push!(pool.fingerprints, cut.fingerprint)
            pool.fingerprint_to_index[cut.fingerprint] = i
        end
    end
    function process_cuts(cuts, select_ratio, cut_type, max_pool_size, pool::CutPool)
        if isempty(relaxation_sol)
            # Sort by violation/norm criteria if no relaxation solution
            @info "No relaxation solution, sorting by violation/norm"
            sorted_cuts = sort(cuts, by=c -> (c.violation / c.norm * sqrt(c.in_round) / (1 + c.depth)), rev=true)
            active_cut_index = 0
        else
            # Sort by activation at relaxation solution
            # @info "Sorting by activation at relaxation solution"
            sorted_cuts = sort(cuts, by=c -> calculate_activation(c.coefficients, relaxation_sol))
            
            # Find largest index where activation > 1e-6
            active_cut_index = 0
            for (i, c) in enumerate(sorted_cuts)
                if calculate_activation(c.coefficients, relaxation_sol) > 1e-9
                    active_cut_index = i
                    break
                end
            end
            # # idx = findlast(c -> calculate_activation(c.coefficients, relaxation_sol) > 1e-9, sorted_cuts)
            #  = idx
        end
        
        # Always include active cuts
        active_cuts = sorted_cuts[1:active_cut_index]
        
        # Sort remaining cuts by violation/norm criteria
        remaining_cuts = sorted_cuts[(active_cut_index+1):end]
        sorted_remaining = sort(remaining_cuts, by=c -> (c.violation / c.norm * sqrt(c.in_round) / (1 + c.depth)), rev=true)

        #  # Trim the global pool to maintain size limit
        # if cut_type == :lazy
        #     glb_pool.lazy_pool.cuts = vcat(active_cuts, sorted_remaining[1:min(end, 1000 - active_cut_index)])
        #     glb_pool.lazy_pool.pool_size = length(glb_pool.lazy_pool.cuts)
        # else
        #     glb_pool.user_pool.cuts = vcat(active_cuts, sorted_remaining[1:min(end, 1000 - active_cut_index)])
        #     glb_pool.user_pool.pool_size = length(glb_pool.user_pool.cuts)
        # end
        # Keep the best cuts up to max_pool_size
        cuts_to_keep = vcat(active_cuts, sorted_remaining[1:min(end, max_pool_size - active_cut_index)])
            
        # Update the pool with kept cuts, maintaining all data structures
        trim_pool!(pool, cuts_to_keep)
        
        # Calculate how many additional cuts to select
        num_additional_select = ceil(Int, length(sorted_remaining) * select_ratio / (current_round^(cut_type == :lazy ? 1/2 : 1/3)))

        # Select additional cuts from remaining (no need to subtract active_cut_index)
        additional_cuts = sorted_remaining[1:min(end, num_additional_select)]
        
        return vcat(active_cuts, additional_cuts), length(cuts), active_cut_index, num_additional_select
    end
    
    # Process both lazy constraints and user cuts
    filtered_lazy_cons, num_lazy_cons_in_pool, num_lazy_active, num_lazy_sel = process_cuts(glb_pool.lazy_pool.cuts, config.lazy_select_ratio, :lazy, max_pool_size, glb_pool.lazy_pool)
    filtered_user_cuts, num_user_cuts_in_pool, num_user_active, num_user_sel = process_cuts(glb_pool.user_pool.cuts, config.user_select_ratio, :user, max_pool_size, glb_pool.user_pool)
    
    # Check pool growth for continuation decision
    is_continue_restart = true
    # if !is_final
    #     lazy_pool_growth = (num_lazy_cons_in_pool - previous_lazy_pool_size) / (previous_lazy_pool_size + 1)
    #     if lazy_pool_growth <= config.lazy_pool_growth_tolerance
    #         @info "Lazy pool growth below $(100*config.lazy_pool_growth_tolerance)%"
    #         is_continue_restart = false
    #     end
    # end

    num_lazy_cons_to_add = num_lazy_active + num_lazy_sel
    num_user_cuts_to_add = num_user_active + num_user_sel
    
    @info "Lazy cons in pool/added: $num_lazy_cons_in_pool / $num_lazy_cons_to_add, active: $num_lazy_active"
    @info "User cuts in pool/added: $num_user_cuts_in_pool / $num_user_cuts_to_add, active: $num_user_active"
    
    return filtered_lazy_cons, filtered_user_cuts, 
            num_lazy_cons_in_pool, num_user_cuts_in_pool,
            num_lazy_cons_to_add, num_user_cuts_to_add,
            num_lazy_active, num_user_active,
            is_continue_restart
end

"""
Run a single instance with specified solver

julia_main(["FlowCutSolver", "0.1", path, ".", "--print-level", "0", "--create-from-scratch", "true", "--max-node-to-tree-restart", "2000", "--restart-rounds", "10"])
args =  ["FlowCutSolver", "0.1", path, ".", "--print-level", "0", "--create-from-scratch", "true", "--max-node-to-tree-restart", "2000", "--tree-restart-rounds", "10"]


path = "data/raw/presolve_test_instance.json"

path = "data/grid/test_20_16/d0.3/f0.3/test_20_16_d0.3_f0.3_se1.json"

path = "data/raw/presolve_test_instance.json"

path = "data/dev2/test_20_50/d0.2/f0.1/test_20_50_d0.2_f0.1_se1.json"
args = ["FlowCutSolver", "0.1", path, "."]
using JuMP
using JSON
import MathOptInterface as MOI
using CPLEX
args =  ["FlowCutSolver", "0.1", path, ".", "--max-time", "10", "--root-restart-rounds", "1", "--tree-restart-rounds", "1"]
using ArgParse
# Parse arguments
s = parse_commandline()
@info "Parsing command line arguments"
parsed_args = parse_args(args, s)
@info "Arguments parsed"
# Create configuration
config = config_from_args(parsed_args)
@info "Configuration created"

instance_path = path
output_dir = "."
raw_data, probs, merge_stats = load_instance_from_json(instance_path)
costs = ones(length(raw_data.edges))

epsilon = 0.3
instance = PPMPInstance(raw_data, probs, costs, epsilon)
solver_type = FlowCutSolver
@info "Starting FlowCutSolver with restart rounds..."


round_solution,               # PPMPSolution: The solution found by the solver at this solve round
final_solution,               # PPMPSolution: final solution found by the solver
solver,                       # PPMPSolver: The optimization solver being used for the instance.
combined_stats,               # SolutionStats: A structure to store combined statistics of the optimization process.
previous_lazy_pool_size,      # Int: The size of the lazy constraint pool from the previous iteration.
glb_pool,                     # FlowCutGlbPool: The global pool of constraints.
incumbent_sol,                # Vector{Float64}: The current best solution found by the solver.
current_obj,                  # Float64: The objective value of the current solution.
filtered_lazy_cons,           # Vector{StoredCut}: The set of lazy constraints that have been filtered.
filtered_user_cuts,           # Vector{StoredCut}: The set of user-defined cuts that have been filtered.
num_lazy_cons_in_pool,        # Int: The number of lazy constraints currently in the global pool.
num_user_cuts_in_pool,        # Int: The number of user-defined cuts currently in the global pool.
num_lazy_cons_to_add,         # Int: The number of lazy constraints to be added in the current solve round.
num_user_cuts_to_add,         # Int: The number of user-defined cuts to be added in the current solve round.
num_lazy_active,              # Int: The number of active lazy constraints in the current solve round.
num_user_active,              # Int: The number of active user-defined cuts in the current solve round.
root_restart_flag,            # Bool: A flag indicating whether a root restart is required.
tree_restart_flag,            # Bool: A flag indicating whether a tree restart is required.
restart_type,                 # String: The type of restart to be performed.
lazy_growth_flag,             # Bool: A flag indicating whether the lazy pool has grown sufficiently to continue restarting.
remain_time,                  # Float64: The remaining time allowed for the optimization process.
node_limit,                   # Int: The limit on the number of nodes to be explored in the optimization process.
target_gap,                   # Float64: The target optimality gap for the optimization process.
is_optimal = init_run_instance(config)

# Restart rounds with lazy pool growth check
# current_round == 1: Initial solve; 
#                > 1: Restart round;
#        out of loop: Final solve
current_round = 1
# # Increase node limit for next round if not setting root restart
node_limit = update_node_limit(node_limit, root_restart_flag, tree_restart_flag, config)
previous_lazy_pool_size = glb_pool.lazy_pool.pool_size

# Solve based on solver type


This is the good version
"""

"""
These below are the restart version
"""

"""
Run instance with restart capability for FlowCutSolver
"""
function run_instance(solver_type::Type{<:PPMPSolver},
                     epsilon::Float64,
                     instance_path::String="data/raw/test_instance.json",
                     output_dir::String="results/raw",
                     config::PPMPConfig=default_config()
                     )
    try
        # Load instance
        @info "Loading instance from $instance_path"
        raw_data, probs, merge_stats = load_instance_from_json(instance_path, merge_scenarios=config.merge_identical_scenarios)
        
        # Create PPMP instance
        costs = ones(length(raw_data.edges))
        # num_edges = length(raw_data.edges)
        # costs = Float64[((i-1) % 5) + 1 for i in 1:num_edges]
        instance = PPMPInstance(raw_data, probs, costs, epsilon)
        
        # Get solver suffix and determine if restart is needed
        suffix = if solver_type == OriginalMIPSolver
            "_OMS"
        elseif solver_type == FlowCutSolver
            "_FCS"
        else
            throw(ArgumentError("Unsupported solver type: $solver_type"))
        end
        
        # Create result base name
        instance_name = replace(basename(instance_path), ".json" => "")
        result_base_name = "$(instance_name)_eps$(epsilon)"

        presolve_info = nothing
        presolve_stats = nothing

        # Apply presolve if configured
        if solver_type == FlowCutSolver && config.is_presolve_flowcut
            instance, presolve_info = ppmp_presolve(instance)
            if presolve_info !== nothing
                presolve_stats = presolve_info.stats
            end
        elseif solver_type == OriginalMIPSolver && config.is_presolve_original
            instance, presolve_info = ppmp_presolve(instance)
            if presolve_info !== nothing
                presolve_stats = presolve_info.stats
            end
        end

        # After presolve
        println("presolve_info: ", presolve_info !== nothing)
        if presolve_info !== nothing
            println("subfixed_scenarios: ", presolve_info.subfixed_scenarios)
        end

        # Solve based on solver type
        if solver_type == FlowCutSolver
            # Initialize variables for solving process control
            round_solution,               # PPMPSolution: The solution found by the solver at this solve round
            final_solution,               # PPMPSolution: final solution found by the solver
            solver,                       # PPMPSolver: The optimization solver being used for the instance.
            combined_stats,               # SolutionStats: A structure to store combined statistics of the optimization process.
            previous_lazy_pool_size,      # Int: The size of the lazy constraint pool from the previous iteration.
            glb_pool,                     # FlowCutGlbPool: The global pool of constraints.
            incumbent_sol,                # Vector{Float64}: The current best solution found by the solver.
            current_obj,                  # Float64: The objective value of the current solution.
            filtered_lazy_cons,           # Vector{StoredCut}: The set of lazy constraints that have been filtered.
            filtered_user_cuts,           # Vector{StoredCut}: The set of user-defined cuts that have been filtered.
            num_lazy_cons_in_pool,        # Int: The number of lazy constraints currently in the global pool.
            num_user_cuts_in_pool,        # Int: The number of user-defined cuts currently in the global pool.
            num_lazy_cons_to_add,         # Int: The number of lazy constraints to be added in the current solve round.
            num_user_cuts_to_add,         # Int: The number of user-defined cuts to be added in the current solve round.
            num_lazy_active,              # Int: The number of active lazy constraints in the current solve round.
            num_user_active,              # Int: The number of active user-defined cuts in the current solve round.
            root_restart_flag,            # Bool: A flag indicating whether a root restart is required.
            tree_restart_flag,            # Bool: A flag indicating whether a tree restart is required.
            restart_type,                 # String: The type of restart to be performed.
            lazy_growth_flag,             # Bool: A flag indicating whether the lazy pool has grown sufficiently to continue restarting.
            remain_time,                  # Float64: The remaining time allowed for the optimization process.
            node_limit,                   # Int: The limit on the number of nodes to be explored in the optimization process.
            target_gap,                   # Float64: The target optimality gap for the optimization process.
            is_optimal = init_run_instance(config)

            # Restart rounds with lazy pool growth check
            # current_round == 1: Initial solve; 
            #                > 1: Restart round;
            #        out of loop: Final solve
            current_round = 1
            # # Increase node limit for next round if not setting root restart
            node_limit = update_node_limit(node_limit, root_restart_flag, tree_restart_flag, config)
            previous_lazy_pool_size = glb_pool.lazy_pool.pool_size

            while root_restart_flag || tree_restart_flag

                # @info "Starting FlowCutSolver with $(config.root_restart_rounds) root restart rounds and $(config.tree_restart_rounds) tree restart rounds..."

                @info "Starting restart round $current_round..."
                if root_restart_flag
                    restart_type = "root"
                    @info "Root restart round $current_round"
                elseif tree_restart_flag
                    restart_type = "tree"
                    @info "Tree restart round $current_round, node_limit: $node_limit"
                end
                
                @info "Node limit for this round: $node_limit"
                # solver settings
                solver = current_round == 1 ?
                     solver_type(instance, config=config, presolve_info=presolve_info, solve_rounds=current_round, gap=target_gap, node_limit=node_limit, time_limit=remain_time) :
                     
                     solver_type(instance, 
                                networks=solver.networks, 
                                config=config, 
                                presolve_info=presolve_info, 
                                incumbent_sol=incumbent_sol,          # warm start
                                stored_lazy_cons=filtered_lazy_cons,  # add lazy cons
                                stored_user_cuts=filtered_user_cuts,  # add user cuts
                                solve_rounds=current_round,           # current round number
                                gap=target_gap,                       # target gap for this round solving
                                node_limit=node_limit,                # node limit for this round solving
                                time_limit=remain_time)               # time limit for this round solving
            
                round_solution = solve(solver, glb_pool) # solve the model

                @info "Round $current_round solve finished"
                # Get incumbent solution and objective from previous round
                incumbent_sol = vcat(value.(solver.x), value.(solver.z))
                # Get best obj_value from previous round
                current_obj = objective_value(solver.model) # this variable not used yet

                @info "Incumbent solution and objective value saved"
                
                

                # Update intermediate round callback statistics
                update_callbackstats_pool(round_solution.stats.callback_stats, glb_pool.lazy_pool.pool_size, glb_pool.user_pool.pool_size, num_lazy_cons_to_add, num_user_cuts_to_add, num_lazy_active, num_user_active, glb_pool.lazy_pool.fingerprint_hit, glb_pool.user_pool.fingerprint_hit)
                
                # Save intermediate results to round-indexed file
                write_round_solve_info(round_solution, output_dir, result_base_name, current_round, restart_type)


                # Update current solution (PPMPSolution) and combine stats
                combined_stats = combined_stats === nothing ? round_solution.stats : merge_solution_stats(combined_stats, round_solution.stats)

                @info "Combined stats updated"
                # Filter lazy cons and user cuts for next round (or final solve), break in loop only happens after this
                filtered_lazy_cons, filtered_user_cuts, num_lazy_cons_in_pool, num_user_cuts_in_pool, num_lazy_cons_to_add, num_user_cuts_to_add, num_lazy_active, num_user_active, lazy_growth_flag = 
                get_filter_lazy_cons_user_cuts(glb_pool, config, previous_lazy_pool_size, current_round, false)
                @info "Lazy cons and user cuts filtered"

                if !lazy_growth_flag
                    if root_restart_flag
                        root_restart_flag = false
                    elseif tree_restart_flag
                        tree_restart_flag = false
                    end
                end

                root_restart_flag, tree_restart_flag = update_restart_flag(root_restart_flag, tree_restart_flag, current_round, config)

                node_limit = update_node_limit(node_limit, root_restart_flag, tree_restart_flag, config)

                previous_lazy_pool_size = glb_pool.lazy_pool.pool_size

                # Update remaining time, check if we reached time limit
                remain_time -= MOI.get(solver.model, MOI.SolveTimeSec())
                if remain_time <= 0.0 
                    println("Time limit reached, stopping solving")
                    is_optimal = true
                    break
                end

                @info "Remaining time for next round: $remain_time"

                # update target gap for next round solving
                previous_gap = round_solution.stats.gap
                target_gap = max(config.mip_gap, previous_gap * config.gap_multi_factor)
                # Check if we reached the mip_gap
                if target_gap <= config.mip_gap + 1e-7
                    println("Optimality gap reached, stopping solving")
                    is_optimal = true
                    break
                end

                println("Previous gap / New gap : $previous_gap / $target_gap")
                
                if round_solution === nothing
                    println("Round $current_round solve failed")
                    current_round += 1 # increase round number
                    continue
                end

                @info "Round $current_round finished"

                current_round += 1 # increase round number
                
            end

            restart_type = "final"

            if round_solution === nothing
                println("All restart rounds failed")
            end
            
            # Final solve with remaining time
            if is_optimal
                println("Reach optimal, no need for final solve")
                final_solution = round_solution
            else 
                @info "Starting final solve..."
                final_solver =  round_solution === nothing ? 
                    solver_type(instance, config=config, presolve_info=presolve_info, solve_rounds=current_round, gap=config.mip_gap, node_limit=config.max_node, time_limit=remain_time) :
                    solver_type(
                        instance,
                        networks=solver.networks, # use previous generated network
                        config=config,
                        presolve_info=presolve_info,
                        restart_flag=false, # final solve
                        incumbent_sol=incumbent_sol,
                        stored_lazy_cons=filtered_lazy_cons,
                        stored_user_cuts=filtered_user_cuts,
                        solve_rounds=current_round, # auto setting final round
                        gap=config.mip_gap,
                        node_limit=config.max_node,
                        time_limit=remain_time
                    )
            
            
                final_solution = solve(final_solver, glb_pool)

                update_callbackstats_pool(final_solution.stats.callback_stats, glb_pool.lazy_pool.pool_size, glb_pool.user_pool.pool_size, num_lazy_cons_to_add, num_user_cuts_to_add, num_lazy_active, num_user_active, glb_pool.lazy_pool.fingerprint_hit, glb_pool.user_pool.fingerprint_hit)
                
                if config.is_logging_round_info
                    write_round_solve_info(final_solution, output_dir, result_base_name, current_round, restart_type)
                end

                # Update solution (PPMPSolution) and combine stats
                combined_stats = combined_stats === nothing ? final_solution.stats : merge_solution_stats(combined_stats, final_solution.stats)

                combined_stats.callback_stats.num_lazy_cons_in_pool = glb_pool.lazy_pool.pool_size
                combined_stats.callback_stats.num_user_cuts_in_pool = glb_pool.user_pool.pool_size
            end
            # Create final solution with combined stats
            solution = PPMPSolution(
                final_solution.instance,
                final_solution.selected_edges,
                final_solution.scenario_values,
                final_solution.objective_value,
                combined_stats
            )
        else
            # Regular solve for OMS
            @info "Creating solver of type $solver_type"
            solver = solver_type(instance, config=config)
            solution = solve(solver)
        end
        
        if solution !== nothing
            # Create result data
            @info "Creating result data..."
            result = create_result_data(solution, result_base_name, string(solver_type), merge_stats=merge_stats, presolve_stats=presolve_stats)
            
            # Save with matching name
            res_file_path = joinpath(output_dir, "$(result_base_name)$(suffix).json")
            mkpath(output_dir)
            open(res_file_path, "w") do f
                JSON.print(f, result, 2)
            end
            
            println("Saving result to $res_file_path")
            return "Result saved in $output_dir as $res_file_path"
        else
            @info "No solution found"
            return false, "No solution found"
        end
        
    catch e
        println("Error in run_instance: ", e)
        println(stacktrace())
        return false, "Error: $e"
    end
end


function init_run_instance(config::PPMPConfig)
    round_solution = nothing
    final_solution = nothing
    solver = nothing
    combined_stats = nothing

    # Initialize variables
    previous_lazy_pool_size = 0

    # Store all lazy constraints across rounds
    glb_pool = FlowCutGlbPool()

    # Store incumbent solution and objective value for warm restart
    incumbent_sol = nothing
    current_obj = nothing

    # Number of lazy/user cuts added in the beginning of new round solve
    filtered_lazy_cons = []
    filtered_user_cuts = []

    num_lazy_cons_in_pool = 0
    num_user_cuts_in_pool = 0
    
    num_lazy_cons_to_add = 0
    num_user_cuts_to_add = 0

    num_lazy_active = 0
    num_user_active = 0

    root_restart_flag = (config.is_root_restart & config.root_restart_rounds > 0) ? true : false
    tree_restart_flag = (config.is_tree_restart & config.tree_restart_rounds > 0) ? true : false

    restart_type = root_restart_flag ? "root" : "tree"

    lazy_growth_flag = true
    
    remain_time = config.max_time
    node_limit = (!root_restart_flag && config.is_tree_restart && config.max_node_to_tree_restart > 0)  ? config.max_node_to_tree_restart : 0 # 0 means only solve the root node

    target_gap = config.mip_gap_init
    
    is_optimal = false

    return round_solution,               # PPMPSolution: The solution found by the solver at this solve round
        final_solution,                  # PPMPSolution: The final solution found by the solver
        solver,                          # PPMPSolver: The optimization solver being used for the instance.
        combined_stats,                  # SolutionStats: A data structure to store combined statistics of the optimization process.
        previous_lazy_pool_size,         # Int: The size of the lazy constraint pool from the previous iteration.
        glb_pool,                        # FlowCutGlbPool: The global pool of constraints.
        incumbent_sol,                   # Vector{Float64}: The current best solution found by the solver.
        current_obj,                     # Float64: The objective value of the current solution.
        filtered_lazy_cons,              # Vector{StoredCut}: The set of lazy constraints that have been filtered.
        filtered_user_cuts,              # Vector{StoredCut}: The set of user-defined cuts that have been filtered.
        num_lazy_cons_in_pool,           # Int: The number of lazy constraints currently in the global pool.
        num_user_cuts_in_pool,           # Int: The number of user-defined cuts currently in the global pool.
        num_lazy_cons_to_add,            # Int: The number of lazy constraints to be added in the current solve round.
        num_user_cuts_to_add,            # Int: The number of user-defined cuts to be added in the current solve round.
        num_lazy_active,                 # Int: The number of active lazy constraints in the current solve round.
        num_user_active,                 # Int: The number of active user-defined cuts in the current solve round.
        root_restart_flag,               # Bool: A flag indicating whether a root restart is required.
        tree_restart_flag,               # Bool: A flag indicating whether a tree restart is required.
        restart_type,                    # String: The type of restart to be performed.
        lazy_growth_flag,                # Bool: A flag indicating whether the lazy pool has grown sufficiently to continue restarting.
        remain_time,                     # Float64: The remaining time allowed for the optimization process.
        node_limit,                      # Int: The limit on the number of nodes to be explored in the optimization process.
        target_gap,                      # Float64: The target optimality gap for the optimization process.
        is_optimal                       # Bool: A flag indicating whether the current round solve is optimal.

end

"""
    update_node_limit(node_limit, root_restart_flag, tree_restart_flag, config)

Update the node limit based on the configuration and restart flags.

# Arguments
- `node_limit::Int`: The current node limit.
- `root_restart_flag::Bool`: Flag indicating if a root restart has occurred.
- `tree_restart_flag::Bool`: Flag indicating if a tree restart has occurred.
- `config::PPMPConfig`: Configuration object containing node limit factors and thresholds.

# Returns
- `Int`: The updated node limit.
"""
function update_node_limit(node_limit, root_restart_flag, tree_restart_flag, config)
    if config.node_limit_factor >= 1.0 && !root_restart_flag && tree_restart_flag
        return min(config.max_node, ceil(Int, max(node_limit* config.node_limit_factor, config.max_node_to_tree_restart)))
    end
    return node_limit
end

"""
    update_restart_flag(root_restart_flag::Bool, tree_restart_flag::Bool, current_round::Int, config::PPMPConfig)

Update the restart flags based on the current round and configuration.

# Arguments
- `root_restart_flag::Bool`: Flag indicating if a root restart has occurred.
- `tree_restart_flag::Bool`: Flag indicating if a tree restart has occurred.
- `current_round::Int`: The current round number.
- `config::PPMPConfig`: Configuration object containing restart round thresholds.

# Returns
- `Tuple{Bool, Bool}`: The updated root and tree restart flags.
"""
function update_restart_flag(root_restart_flag::Bool, tree_restart_flag::Bool, current_round::Int, config::PPMPConfig)
    root_restart_flag = current_round >= config.root_restart_rounds ? false : root_restart_flag
    tree_restart_flag = current_round - config.root_restart_rounds >= config.tree_restart_rounds ? false : tree_restart_flag
    return root_restart_flag, tree_restart_flag
end

"""
    update_callbackstats_pool(callback_stats::CallbackStats, 
                             num_lazy_cons_in_pool::Int, num_user_cuts_in_pool::Int, 
                             num_lazy_cons_to_add::Int, num_user_cuts_to_add::Int, 
                             num_lazy_active::Int, num_user_active::Int,
                             lazy_fingerprint_hit::Int, user_fingerprint_hit::Int)

Update the callback statistics with the provided values.

# Arguments
- `callback_stats::CallbackStats`: The callback statistics object to update.
- `num_lazy_cons_in_pool::Int`: Number of lazy constraints in the pool.
- `num_user_cuts_in_pool::Int`: Number of user cuts in the pool.
- `num_lazy_cons_to_add::Int`: Number of lazy constraints to add.
- `num_user_cuts_to_add::Int`: Number of user cuts to add.
- `num_lazy_active::Int`: Number of active lazy constraints.
- `num_user_active::Int`: Number of active user cuts.
- `lazy_fingerprint_hit::Int`: Number of lazy fingerprint hits.
- `user_fingerprint_hit::Int`: Number of user fingerprint hits.
"""
function update_callbackstats_pool(callback_stats::CallbackStats, 
                                    num_lazy_cons_in_pool::Int, num_user_cuts_in_pool::Int, 
                                    num_lazy_cons_to_add::Int, num_user_cuts_to_add::Int, 
                                    num_lazy_active::Int, num_user_active::Int,
                                    lazy_fingerprint_hit::Int, user_fingerprint_hit::Int
                                    )
    # number of lazy/user cuts in global pool
    callback_stats.num_lazy_cons_in_pool = num_lazy_cons_in_pool
    callback_stats.num_user_cuts_in_pool = num_user_cuts_in_pool
    # number of lazy/user cuts added
    callback_stats.num_lazy_cons_to_add = num_lazy_cons_to_add
    callback_stats.num_user_cuts_to_add = num_user_cuts_to_add

    # number of active lazy/user added
    callback_stats.num_lazy_active = num_lazy_active
    callback_stats.num_user_active = num_user_active

    callback_stats.lazy_fingerprint_hit = lazy_fingerprint_hit
    callback_stats.user_fingerprint_hit = user_fingerprint_hit

end

"""
    merge_solution_stats(combined_stats::SolutionStats, new_stats::SolutionStats)

Merge two sets of solution statistics into one.

# Arguments
- `combined_stats::SolutionStats`: The combined solution statistics.
- `new_stats::SolutionStats`: The new solution statistics to merge.

# Returns
- `SolutionStats`: The merged solution statistics.
"""
function merge_solution_stats(combined_stats::SolutionStats, new_stats::SolutionStats)
    return SolutionStats(
        combined_stats.solve_time + new_stats.solve_time,
        combined_stats.setup_time + new_stats.setup_time,
        combined_stats.node_count + new_stats.node_count,
        new_stats.obj_value,
        new_stats.obj_bound,
        new_stats.gap,
        new_stats.status,
        merge_callback_stats(combined_stats.callback_stats, new_stats.callback_stats)
    )
end

"""
    merge_callback_stats(initial_stats::CallbackStats, restart_stats::CallbackStats)

Merge callback statistics from two solves.

# Arguments
- `initial_stats::CallbackStats`: The initial callback statistics.
- `restart_stats::CallbackStats`: The callback statistics from a restart.

# Returns
- `CallbackStats`: The merged callback statistics.
"""
function merge_callback_stats(initial_stats::CallbackStats, restart_stats::CallbackStats)
    merged = CallbackStats()
    
    # Merge counters
    merged.total_cuts_added = initial_stats.total_cuts_added + restart_stats.total_cuts_added
    merged.total_callbacks = initial_stats.total_callbacks + restart_stats.total_callbacks
    merged.user_callbacks = initial_stats.user_callbacks + restart_stats.user_callbacks
    merged.lazy_callbacks = initial_stats.lazy_callbacks + restart_stats.lazy_callbacks
    
    # Merge times
    merged.total_callbacks_time = initial_stats.total_callbacks_time + restart_stats.total_callbacks_time
    merged.user_callbacks_time = initial_stats.user_callbacks_time + restart_stats.user_callbacks_time
    merged.lazy_callbacks_time = initial_stats.lazy_callbacks_time + restart_stats.lazy_callbacks_time
    
    # Merge cuts per scenario
    for (k, v) in initial_stats.cuts_per_scenario
        merged.cuts_per_scenario[k] = get(merged.cuts_per_scenario, k, 0) + v
    end
    for (k, v) in restart_stats.cuts_per_scenario
        merged.cuts_per_scenario[k] = get(merged.cuts_per_scenario, k, 0) + v
    end
    
    # Merge root stats
    merged.root_stats["user_cuts_added"] = initial_stats.root_stats["user_cuts_added"] + 
                                          restart_stats.root_stats["user_cuts_added"]
    merged.root_stats["lazy_cuts_added"] = initial_stats.root_stats["lazy_cuts_added"] + 
                                          restart_stats.root_stats["lazy_cuts_added"]
    merged.root_stats["max_violation"] = max(initial_stats.root_stats["max_violation"],
                                           restart_stats.root_stats["max_violation"])
    
    # Merge tree stats
    merged.tree_stats["user_cuts_added"] = initial_stats.tree_stats["user_cuts_added"] + 
                                          restart_stats.tree_stats["user_cuts_added"]
    merged.tree_stats["lazy_cuts_added"] = initial_stats.tree_stats["lazy_cuts_added"] + 
                                          restart_stats.tree_stats["lazy_cuts_added"]
    merged.tree_stats["nodes_processed"] = initial_stats.tree_stats["nodes_processed"] + 
                                          restart_stats.tree_stats["nodes_processed"]
    
    # Set restart-specific stats
    merged.is_restart = true
    
    return merged
end

"""
#       writes the solution information of a specific round to a JSON file.
# 
# Arguments
- `output_dir::String`: The directory where the result file will be saved.
- `result_base_name::String`: The base name for the result file.
- `current_round::Int`: The current round number.
- `restart_type::String`: The type of restart used.
# 
The function creates a result data structure, constructs the file path, ensures the output directory exists,
writes the result data to a JSON file.
"""
function write_round_solve_info(solution::PPMPSolution, output_dir::String, result_base_name::String, current_round::Int, restart_type::String)
    round_result = create_result_data(solution, "$(result_base_name)_$(restart_type)$(current_round)", "FlowCutSolver")
    round_file_path = joinpath(output_dir, "$(result_base_name)_$(restart_type)$(current_round)_FCS.json")
    mkpath(output_dir)
    open(round_file_path, "w") do f
        JSON.print(f, round_result, 2)
    end
    @info "$(restart_type) $(current_round) result saved to $round_file_path"
end