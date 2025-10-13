# src/algorithms/base.jl

using Graphs
using SimpleWeightedGraphs
using MinCostFlows

# Base algorithm implementations and abstract types

"""
Abstract type for all PPMP solvers
"""
abstract type PPMPSolver end

"""
Abstract type for all cutting plane generators
"""
abstract type CutGenerator end

"""
Base functionality for all solvers
"""
abstract type AbstractSolver end

"""
Create the flow network for a scenario
Returns a vector of arcs (i,j) where:
- 1:n are left nodes
- n+1:2n are right nodes
- 2n+1 is source (s)
- 2n+2 is sink (t)
"""
function create_flow_network(n::Int, edges::Vector{Tuple{Int,Int}})
    # Create node sets: L (1:n), R (n+1:2n), s (2n+1), t (2n+2)
    s, t = 2n + 1, 2n + 2
    
    # Create arc set for the flow network
    arcs = Vector{Tuple{Int,Int}}()
    
    # Add source to L arcs
    append!(arcs, [(s, i) for i in 1:n])
    
    # Add R to sink arcs
    append!(arcs, [(n+i, t) for i in 1:n])
    
    # Add L to R arcs (from original edges)
    append!(arcs, [(i, n+j) for (i,j) in edges])
    
    # Add sink to source arc
    push!(arcs, (t, s))
    
    return arcs
end


"""
Network configuration for scenario-based flow problems
"""
struct FlowNetwork
    graph::SimpleWeightedDiGraph{Int,Float64}
    source::Int
    sink::Int
    left_nodes::UnitRange{Int}
    right_nodes::UnitRange{Int}
    lr_arcs::Vector{Tuple{Int,Int}}  # L->R arcs in this network
end

"""
Create capacitated flow network for cut generation
"""
function create_flow_network_capacitated(n::Int, scenario_edges::Vector{Tuple{Int,Int}})
    num_vertices = 2n + 2
    s, t = 2n + 1, 2n + 2
    
    # Create arrays for one-shot graph creation
    sources = Int[]
    destinations = Int[]
    weights = Float64[]
    
    # Add s->L arcs
    append!(sources, fill(s, n))
    append!(destinations, 1:n)
    append!(weights, ones(n))
    
    # Add R->t arcs
    append!(sources, (n+1):2n)
    append!(destinations, fill(t, n))
    append!(weights, ones(n))
    
    # Add L->R arcs for this scenario
    lr_arcs = Tuple{Int,Int}[]
    for (i,j) in scenario_edges
        push!(sources, i)
        push!(destinations, n+j)
        push!(weights, 1.0)
        push!(lr_arcs, (i,n+j))
    end
    
    # Create the network in one shot
    graph = SimpleWeightedDiGraph(sources, destinations, weights)
    
    return FlowNetwork(graph, s, t, 1:n, (n+1):2n, lr_arcs)
end

"""
Create capacitated flow network with initial costs for mixing cut generation. (Using MinCostFlows)
"""
function create_flow_network_capacitated_costed(n::Int, scenario_edges::Vector{Tuple{Int,Int}})
    num_vertices = 2n + 2
    s, t = 2n + 1, 2n + 2
    
    # Create arrays for one-shot FlowProblem creation
    nodefrom = Int[]
    nodeto = Int[]
    limit = Int[]
    cost = Int[]
    
    # Add s->L arcs
    append!(nodefrom, fill(s, n))
    append!(nodeto, 1:n)
    append!(limit, ones(Int, n))
    append!(cost, zeros(Int, n))
    
    # Add R->t arcs
    append!(nodefrom, (n+1):2n)
    append!(nodeto, fill(t, n))
    append!(limit, ones(Int, n))
    append!(cost, zeros(Int, n))
    
    # Add L->R arcs for this scenario
    for (i,j) in scenario_edges
        push!(nodefrom, i)
        push!(nodeto, n+j)
        push!(limit, 1)
        push!(cost, 0)
    end
    
    # Set up injection vector
    injection = zeros(Int, num_vertices)
    injection[s] = n
    injection[t] = -n
    
    return FlowProblem(nodefrom, nodeto, limit, cost, injection)
end

"""
Update network capacities efficiently for a given scenario and solution values
Note: weights matrix is indexed by (dst, src)
"""
function update_network_capacities!(network::FlowNetwork, 
                                  x_val::Vector{Float64},
                                  z_val::Float64,
                                  edge_to_index::Dict{Tuple{Int,Int}, Int})
    n = length(network.left_nodes)
    s, t = network.source, network.sink
    weights = network.graph.weights
    
    # Update s->L capacities to z_val (weights[dst,src])
    for i in network.left_nodes
        weights[i,s] = z_val
    end
    
    # Update R->t capacities to z_val (weights[dst,src])
    for i in network.right_nodes
        weights[t,i] = z_val
    end
    
    # Update L->R edge capacities based on x_val (weights[dst,src])
    for (i,j) in network.lr_arcs
        orig_j = j - n  # recover original right node index
        edge_idx = edge_to_index[(i,orig_j)]
        weights[j,i] = x_val[edge_idx]  # Note the reverse indexing
    end
end

"""
Print detailed model information to a file
"""
function print_model_details(solver::AbstractSolver, filename::String="model_details.out")
    open(filename, "w") do f
        println(f, "Model Summary")
        println(f, "=============")
        println(f, "\nNumber of variables: ", num_variables(solver.model))
        
        println(f, "\nVariables:")
        println(f, "=========")
        for var in all_variables(solver.model)
            println(f, "$(name(var)): [$(lower_bound(var)), $(upper_bound(var))]")
        end
        
        println(f, "\nConstraints:")
        println(f, "===========")
        for (idx, con) in enumerate(all_constraints(solver.model, include_variable_in_set_constraints=false))
            println(f, "c$idx: $(name(con))")
            println(f, "  ", con)
        end
    end
end

"""
Write model to MPS file and generate CPLEX command script
"""
function export_to_mps(solver::PPMPSolver,
                      mps_filename::String="model.mps",
                      cplex_script::String="solve.cplex")
    # Write MPS file
    write_to_file(solver.model, mps_filename)
    
    # Generate CPLEX command file
    # open(cplex_script, "w") do f
    #     println(f, "# CPLEX commands to solve $(mps_filename)")
    #     println(f, "# Run with: cplex < $(cplex_script)")
    #     println(f, "read $(mps_filename)")
    #     println(f, "optimize")
    #     println(f, "display solution objective")
    #     println(f, "display solution variables -")
    #     println(f, "write solution.sol")
    #     println(f, "quit")
    # end
    
    # Print instructions
    println("\nFiles generated:")
    println("$(mps_filename) - MPS format model file")
    # println("2. $(cplex_script) - CPLEX commands script")
    # println("\nTo solve with CPLEX, run either:")
    println("\nTo solve with CPLEX, run:")
    # println("   cplex < $(cplex_script)")
    # println("   or")
    println("   cplex -c \"read $(mps_filename)\" \"optimize\" \"display solution variables -\" \"quit\"")
end

"""
Convenience function to both print details and export MPS
"""
function write_model(solver::AbstractSolver, 
                    details_filename::String="model_details.out",
                    mps_filename::String="model.mps",
                    cplex_script::String="solve.cplex")
    print_model_details(solver, details_filename)
    export_to_mps(solver, mps_filename, cplex_script)
end


"""
Validate solution using original MIP formulation
"""
function validate_solution(solution::PPMPSolution)
    @info "Validating solution..."
    start_time = time()
    
    try
        # Create original MIP solver for validation
        validator = OriginalMIPSolver(solution.instance)
        
        # Fix x and z variables to the solution values
        for (i, val) in enumerate(solution.selected_edges)
            fix(validator.x[val], 1; force=true)
        end
        for i in 1:length(validator.x)
            if i ∉ solution.selected_edges
                fix(validator.x[i], 0; force=true)
            end
        end
        
        for (i, val) in enumerate(solution.scenario_values)
            fix(validator.z[i], val ? 1.0 : 0.0; force=true)
        end
        
        # Set solver parameters for validation
        set_optimizer_attribute(validator.model, "CPX_PARAM_EPGAP", 1e-9)  # Tight tolerance
        set_optimizer_attribute(validator.model, "CPX_PARAM_THREADS", 1)   # Single thread
        set_silent(validator.model)  # Suppress output
        
        # Try to solve the validation model
        optimize!(validator.model)
        
        # Check results
        status = termination_status(validator.model)
        is_feasible = status == MOI.OPTIMAL
        validation_obj = is_feasible ? objective_value(validator.model) : Inf
        
        # Detailed validation results
        validation_result = Dict{String,Any}(
            "status" => status,
            "is_feasible" => is_feasible,
            "validation_obj" => validation_obj,
            "original_obj" => solution.objective_value,
            "obj_difference" => abs(validation_obj - solution.objective_value),
            "validation_time" => time() - start_time
        )
        
        # Check flow feasibility for each scenario if solution is feasible
        if is_feasible
            flow_violations = Dict{Int,Float64}()
            max_flow_violation = 0.0
            
            for k in 1:length(solution.instance.scenarios)
                if solution.scenario_values[k]  # Only check selected scenarios
                    # Get flow values
                    flow_vals = Dict{Tuple{Int,Int},Float64}()
                    for (arc, var) in validator.y[k]
                        flow_vals[arc] = value(var)
                    end
                    
                    # Check flow conservation
                    n = solution.instance.n
                    s, t = 2n + 1, 2n + 2
                    flow_val = flow_vals[(t,s)]
                    required_flow = n * value(validator.z[k])
                    violation = required_flow - flow_val
                    
                    flow_violations[k] = violation
                    max_flow_violation = max(max_flow_violation, violation)
                end
            end
            
            validation_result["flow_violations"] = flow_violations
            validation_result["max_flow_violation"] = max_flow_violation
            validation_result["probability_sum"] = sum(solution.instance.probabilities[k] 
                                                     for k in 1:length(solution.scenario_values) 
                                                     if solution.scenario_values[k])
        end
        
        # Print validation results
        println("\nSolution Validation Results:")
        println("==========================")
        println("Feasibility: ", is_feasible)
        println("Validation Status: ", status)
        if is_feasible
            println("Original Objective: ", round(solution.objective_value, digits=6))
            println("Validation Objective: ", round(validation_obj, digits=6))
            println("Objective Difference: ", 
                    round(abs(validation_obj - solution.objective_value), digits=8))
            println("Maximum Flow Violation: ", 
                    round(validation_result["max_flow_violation"], digits=8))
            println("Selected Scenarios Probability Sum: ", 
                    round(validation_result["probability_sum"], digits=6))
            
            # Print detailed flow violations if any significant violations exist
            if any(v -> v > 1e-6, values(validation_result["flow_violations"]))
                println("\nFlow Violations by Scenario:")
                for (k, violation) in validation_result["flow_violations"]
                    if violation > 1e-6
                        println("Scenario $k: ", round(violation, digits=8))
                    end
                end
            end
        end
        
        return validation_result
        
    catch e
        @error "Validation failed" exception=(e, catch_backtrace())
        return Dict{String,Any}(
            "status" => "ERROR",
            "is_feasible" => false,
            "error" => e
        )
    end
end