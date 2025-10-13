# src/types/solution_types.jl


"""
Solution statistics for tracking solver performance
"""
mutable struct SolutionStats
    solve_time::Float64      # Solver runtime
    setup_time::Float64      # Model setup time
    node_count::Int         # Number of branch-and-bound nodes
    obj_value::Float64      # Best objective value found
    obj_bound::Float64      # Best bound on objective
    gap::Float64            # Optimality gap
    status::Any             # Solution status (using MOI status types)
    callback_stats::Union{Nothing,CallbackStats}  # Optional callback statistics
    
    function SolutionStats(solve_time::Float64, setup_time::Float64, 
                          node_count::Int, obj_value::Float64,
                          obj_bound::Float64, gap::Float64, status::Any,
                          callback_stats::Union{Nothing,CallbackStats}=nothing)
        new(solve_time, setup_time, node_count, obj_value, 
            obj_bound, gap, status, callback_stats)
    end
end

"""
Complete solution output
"""
struct PPMPSolution
    instance::PPMPInstance
    selected_edges::Vector{Int}      # Selected edge indices
    scenario_values::Vector{Bool}    # Which scenarios are satisfied
    objective_value::Float64         # Solution objective value
    stats::SolutionStats            # Solution statistics
end

"""
Represents a cut generated during the solution process
"""
struct Cut
    coefficients::Vector{Float64}
    rhs::Float64
    type::Symbol
    violation::Float64
    norm::Float64 # 2_Norm of the cut coefficients
    k_idx::Int # Scenario index
end

"""
Represents the progress of the solution process
"""
mutable struct SolutionProgress
    time_stamps::Vector{Float64}
    objectives::Vector{Float64}
    bounds::Vector{Float64}
    gaps::Vector{Float64}
end

"""
Print detailed solution statistics
"""
function print_solution_stats(solution::PPMPSolution)
    println("\nSolution Statistics:")
    println("===================")
    println("Objective Value: ", solution.objective_value)
    println("Solution Time: ", round(solution.stats.solve_time, digits=2), " seconds")
    println("Setup Time: ", round(solution.stats.setup_time, digits=2), " seconds")
    println("Node Count: ", solution.stats.node_count)
    println("Optimality Gap: ", round(solution.stats.gap * 100, digits=2), "%")
    println("Status: ", solution.stats.status)
    
    # Print callback statistics if available
    if !isnothing(solution.stats.callback_stats)
        println("\nCallback Performance:")
        println("====================")
        cb_stats = solution.stats.callback_stats
        println("Total Callbacks: ", cb_stats.total_callbacks)
        println("Total Time: ", round(cb_stats.total_callbacks_time, digits=2), " seconds")
        println("User Callbacks: ", cb_stats.user_callbacks)
        println("Lazy Callbacks: ", cb_stats.lazy_callbacks)
        println("Total Cuts Added: ", cb_stats.total_cuts_added)
        
        # Root node statistics
        println("\nRoot Node Statistics:")
        println("-------------------")
        println("User Cuts Added: ", cb_stats.root_stats["user_cuts_added"])
        println("Lazy Cuts Added: ", cb_stats.root_stats["lazy_cuts_added"])
        if haskey(cb_stats.root_stats, "max_violation")
            println("Maximum Violation: ", round(cb_stats.root_stats["max_violation"], digits=6))
        end
        
        # Tree statistics
        println("\nTree Node Statistics:")
        println("-------------------")
        println("User Cuts Added: ", cb_stats.tree_stats["user_cuts_added"])
        println("Lazy Cuts Added: ", cb_stats.tree_stats["lazy_cuts_added"])
        println("Nodes Processed: ", cb_stats.tree_stats["nodes_processed"])
        
        # Cut violation analysis
        if !isempty(cb_stats.cut_violations)
            println("\nCut Violation Analysis:")
            println("---------------------")
            println("Maximum Violation: ", round(maximum(cb_stats.cut_violations), digits=6))
            println("Average Violation: ", round(mean(cb_stats.cut_violations), digits=6))
            println("Median Violation: ", round(median(cb_stats.cut_violations), digits=6))
        end
        
        # Scenario analysis
        if !isempty(cb_stats.cuts_per_scenario)
            println("\nScenario Analysis:")
            println("-----------------")
            scenario_cuts = sort(collect(cb_stats.cuts_per_scenario))
            total_scenarios = length(scenario_cuts)
            if total_scenarios > 10
                # Print summary for large number of scenarios
                println("Total Scenarios with Cuts: ", total_scenarios)
                println("Top 5 Scenarios by Cut Count:")
                for (k, count) in scenario_cuts[end-4:end]
                    println("  Scenario $k: $count cuts")
                end
            else
                # Print all scenarios for small number
                println("Cuts per Scenario:")
                for (k, count) in scenario_cuts
                    println("  Scenario $k: $count cuts")
                end
            end
        end
    end
    
    # Print solution structure
    println("\nSolution Structure:")
    println("==================")
    println("Selected Edges: ", length(solution.selected_edges))
    println("Satisfied Scenarios: ", count(solution.scenario_values))
    
    if length(solution.selected_edges) ≤ 20
        println("\nSelected Edges:")
        for edge_idx in solution.selected_edges
            println("  Edge $edge_idx")
        end
    end
end