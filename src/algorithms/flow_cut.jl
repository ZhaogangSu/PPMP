# src/algorithms/flow_cut.jl

using Graphs
using SimpleWeightedGraphs
using GraphsFlows
using SparseArrays
using DataStructures
using Random
using JuMP
using Printf

"""
Solver using the flow-cut formulation with branch-and-Benders approach
"""
struct FlowCutSolver <: PPMPSolver
    model::Model
    instance::PPMPInstance
    x::Vector{VariableRef}    # Edge selection variables
    z::Vector{VariableRef}    # Scenario selection variables
    num_x::Int               # Number of x variables (edges)
    num_z::Int               # Number of z variables (scenarios)
    setup_time::Float64
    networks::Vector{FlowNetwork}  # Scenario-specific networks
    cost_networks::Vector{FlowProblem}  # Scenario-specific cost networks, using struct in MinCostFlows.jl
    config::PPMPConfig      # Solver configuration

    cutpool::CutPool                # Pool of added, mainly for dupilicity check

    # For statistics
    stats::CallbackStats       # Callback statistics for this solver

    # For restart
    restart_flag::Bool        # Flag for restart: false means final solve
    incumbent_sol::Vector{Float64}  # Incumbent solution values
    # best_primal_bound::Float64  # Best primal bound found
    stored_lazy_cons::Vector{StoredCut} # Stored lazy constraints
    stored_user_cuts::Vector{StoredCut} # Stored user cuts
    solve_rounds::Int        # Number of solve rounds
    gap::Float64            # Current relative gap
    node_limit::Int         # Node limit for current solve
    time_limit::Float64         # Time limit for current solve
    presolve_info::Union{Nothing,PresolveInfo}  # Presolve information
end

"""
Initialize flow-cut solver with basic model structure and initial cuts
"""
function FlowCutSolver(instance::PPMPInstance;
                    # fixed_scenarios::Vector{Int}=Int[],
                    networks::Vector{FlowNetwork}=FlowNetwork[],
                    config::PPMPConfig=PPMPConfig(),
                    cutpool::CutPool=CutPool(),
                    restart_flag::Bool=true,
                    presolve_info::Union{Nothing,PresolveInfo}=nothing,  # Add presolve_info parameter
                    incumbent_sol::Vector{Float64}=Float64[],
                    # best_primal_bound::Float64=Inf,
                    stored_lazy_cons::Vector{StoredCut}=StoredCut[],
                    stored_user_cuts::Vector{StoredCut}=StoredCut[],
                    solve_rounds::Int=1,
                    gap::Float64=1e-6,
                    node_limit::Int=1000000,
                    time_limit::Float64=14400.0
                    )
    
    start_time = time()
    @info "Creating CPLEX optimizer..."
    model = Model(CPLEX.Optimizer)
    @info "CPLEX optimizer Successfully created"
    
    # Create variables and basic constraints
    x = @variable(model, [i=1:length(instance.edges)], Bin, base_name="x")
    z = @variable(model, [i=1:length(instance.scenarios)], Bin, base_name="z")
    
    @objective(model, Min, sum(instance.costs .* x))
    
    # Add chance constraint
    @constraint(model, sum(instance.probabilities .* z) >= 1 - instance.epsilon,
               base_name="chance_cons")
    
   # Fix variables for both fixed and subfixed scenarios from presolve
   if config.is_presolve_fix_scenario && presolve_info !== nothing
        # Fix epsilon-fixed scenarios
        for k in presolve_info.fixed_scenarios
            @constraint(model, z[k] == 1, base_name="fixed_z_$k")
        end
        
        # Fix epsilon-subfixed scenarios (previously removed, now fixed to 1)
        for k in presolve_info.subfixed_scenarios
            @constraint(model, z[k] == 1, base_name="subfixed_z_$k")
        end
    end
        
    if solve_rounds > 1 # Restart from stored cuts
        # Add stored lazy constraints if this is a restart
        if !isempty(stored_lazy_cons)
            @info "Adding $(length(stored_lazy_cons)) stored lazy constraints..."
            for (i, con) in enumerate(stored_lazy_cons)
                vars = vcat(x, z)
                @constraint(model, 
                sum(con.coefficients[i] * vars[i] for i in 1:length(vars)) >= 0,
                base_name="stored_lazy_$(i)")
            end
        end
        
        # Add stored lazy constraints if this is a restart
        if !isempty(stored_user_cuts)
            
            @info "Adding $(length(stored_user_cuts)) stored user cuts..."
            for (i, con) in enumerate(stored_user_cuts)
                vars = vcat(x, z)
                @constraint(model, 
                sum(con.coefficients[i] * vars[i] for i in 1:length(vars)) >= 0,
                base_name="stored_lazy_$(i)")
            end
        end
    
    else # solve_rounds == 1  Not a restart
        n = instance.n # Number of one part vertices

        if !config.create_from_scratch
            @info "Adding initial constraints for each scenario..."
            for k in 1:length(instance.scenarios)
                scenario_edges = instance.scenarios[k]
                
                # Add simple degree constraints for each vertex
                for i in 1:n
                    
                    if config.create_left_cons
                        # Edges incident to left vertex i in this scenario
                        left_edges = [(i,j) for (l,j) in scenario_edges if l == i]
                        if !isempty(left_edges)
                            @constraint(model, 
                                sum(x[instance.edge_to_index[e]] for e in left_edges) >= z[k],
                                base_name="left_degree_$(k)_$(i)")
                        end
                    end
                    
                    if config.create_right_cons
                        # # Edges incident to right vertex i in this scenario
                        right_edges = [(j,i) for (j,r) in scenario_edges if r == i]
                        if !isempty(right_edges)
                            @constraint(model, 
                                sum(x[instance.edge_to_index[e]] for e in right_edges) >= z[k],
                                base_name="right_degree_$(k)_$(i)")
                        end
                    end
                end

                if config.create_random_cons
                    # Generate n different cuts using deterministic sampling
                    for seed in 1:n
                        # Use seed for deterministic randomization
                        rng = MersenneTwister(seed)
                        
                        # First cut: |M| = n/2+1, |N| = n/2
                        M = sort(shuffle(rng, 1:n)[1:div(3*n+1,4)])  # 3n/4+1 left vertices
                        N = sort(shuffle(rng, 1:n)[1:div(3*n,4)])    # 3n/4 right vertices
                        
                        # Add cut for (M,N)
                        relevant_edges = [(i,j) for (i,j) in scenario_edges if i in M && j in N]
                        if !isempty(relevant_edges)
                            @constraint(model,
                                sum(x[instance.edge_to_index[e]] for e in relevant_edges) >= 
                                    (length(M) + length(N) - n) * z[k],
                                base_name="init_cut_$(k)_$(seed)a")
                        end
                        
                        # Second cut: |M| = n/2, |N| = n/2+1
                        M = sort(shuffle(rng, 1:n)[1:div(n,2)])    # n/2 left vertices
                        N = sort(shuffle(rng, 1:n)[1:div(n+1,2)])  # n/2+1 right vertices
                        
                        # Add cut for (M,N)
                        relevant_edges = [(i,j) for (i,j) in scenario_edges if i in M && j in N]
                        if !isempty(relevant_edges)
                            @constraint(model,
                                sum(x[instance.edge_to_index[e]] for e in relevant_edges) >= 
                                    (length(M) + length(N) - n) * z[k],
                                base_name="init_cut_$(k)_$(seed)b")
                        end
                    end
                end
            end
        end

        # # Build scenario-specific networks using failed edges
        # if isempty(networks)
        #     @info "Creating flow networks..."
        #     networks = Vector{FlowNetwork}(undef, length(instance.scenarios))
        #     for k in 1:length(instance.scenarios)
        #         networks[k] = create_flow_network_capacitated(
        #             n,
        #             instance.scenarios[k]
        #         )
        #     end
        # end

        # Build scenario-specific networks for both regular flow and min cost flow
        if isempty(networks)
            @info "Creating flow networks..."
            networks = Vector{FlowNetwork}(undef, length(instance.scenarios))
            cost_networks = Vector{FlowProblem}(undef, length(instance.scenarios))
            
            for k in 1:length(instance.scenarios)
                networks[k] = create_flow_network_capacitated(
                    n,
                    instance.scenarios[k]
                )
            end

            @info "Creating cost flow networks..."
            for k in 1:length(instance.scenarios)
                cost_networks[k] = create_flow_network_capacitated_costed(
                    n,
                    instance.scenarios[k]
                )
            end
        end
    end    

    setup_time = time() - start_time
    @info "Creating FlowCutSolver object..."
    return FlowCutSolver(model, instance, x, z, length(instance.edges), 
                        length(instance.scenarios), setup_time, networks, cost_networks, config,
                        cutpool,
                        CallbackStats(),
                        restart_flag,
                        incumbent_sol, 
                        # best_primal_bound, 
                        stored_lazy_cons, stored_user_cuts, solve_rounds, gap, node_limit, time_limit,
                        presolve_info)  # Add presolve_info to the struct
end


"""
Solve PPMP instance using flow-cut formulation with callbacks
"""
function solve(solver::FlowCutSolver, glb_pool::FlowCutGlbPool)
    config = solver.config # Solver configuration

    try
        # output solver.model's constraints and variables number
        # println("number of constraints: ", JuMP.num_constraints(solver.model))
        # println("number of variables: ", JuMP.num_variables(solver.model))
        # println("number of x variables: ", solver.num_x)
        # println("number of z variables: ", solver.num_z)
        # println("number of scenarios: ", length(solver.instance.scenarios))
        # println("number of edges: ", length(solver.instance.edges))
        # println("number of nodes: ", solver.instance.n)

        
        # Set CPLEX parameters
        set_optimizer_attribute(solver.model, "CPX_PARAM_THREADS", config.threads)  # Number of threads
        
        set_optimizer_attribute(solver.model, "CPX_PARAM_TILIM", solver.time_limit)  # Time limit
        set_optimizer_attribute(solver.model, "CPX_PARAM_NODELIM", solver.node_limit)
        set_optimizer_attribute(solver.model, "CPX_PARAM_EPGAP", solver.gap)

        # Set incumbent solution for restart rounds
        if solver.solve_rounds > 1 && !isempty(solver.incumbent_sol)
            num_edges = length(solver.instance.edges)
            MOI.set.(solver.model, MOI.VariablePrimalStart(), solver.x, 
                    solver.incumbent_sol[1:num_edges])
            MOI.set.(solver.model, MOI.VariablePrimalStart(), solver.z, 
                    solver.incumbent_sol[num_edges+1:end])
        end

     
        # Enable cuts and lazy constraints
        set_optimizer_attribute(solver.model, "CPX_PARAM_PREIND", config.cpx_presolve)  # Disable presolve for callbacks
        set_optimizer_attribute(solver.model, "CPXPARAM_MIP_Strategy_CallbackReducedLP", 0)  # 
        # set_optimizer_attribute(solver.model, "CPX_PARAM_MIPCBREDLP", 0)  


        # Memory management parameters
        set_optimizer_attribute(solver.model, "CPX_PARAM_WORKMEM", config.max_memory_mb)
        # set_optimizer_attribute(solver.model, "CPX_PARAM_NODEFILEIND", config.node_file_strategy)

        # Create and set callback
        # glb_pool = FlowCutGlbPool(solver, config)
        # MOI.set(solver.model, CPLEX.CallbackFunction(), 
        #         (cb_data, context_id) -> generate_flow_cuts_callback(cb_data, context_id, cb))

        
        function my_user_callback(cb_data)
            # println("Callback user called outer") 
            return flow_user_cuts_callback(cb_data, solver, config, glb_pool)
        end
        
        function my_lazy_callback(cb_data)
            return flow_lazy_cons_callback(cb_data, solver, config, glb_pool)
        end

        # user cuts callback set up
        if config.user_callback_open
            MOI.set(solver.model, MOI.UserCutCallback(), my_user_callback)
        end
        
        # lazy constraints callback set up
        MOI.set(solver.model, MOI.LazyConstraintCallback(), my_lazy_callback)
        

        # set_attribute(solver.model, MOI.LazyConstraintCallback(), flow_lazy_cons_callback)
        # set_attribute(solver.model, MOI.UserCutCallback(), flow_user_cuts_callback)





        # MOI.set(solver.model, MOI.LazyConstraintCallback(),
        #         (cb_data, context_id) -> flow_lazy_cons_callback(cb_data, context_id, cb))


        # heuristic_callback(cb_data, context_id) = 
                    # generate_heuristic_callback(cb_data, context_id, solver, config)
        # set_attribute(model, MOI.HeuristicCallback(), heuristic_callback)
        # MOI.set(solver.model, CPLEX.CallbackFunction(), 
        #     (cb_data, context_id) -> generate_heuristic_callback(cb_data, context_id, solver, config))
        

        
        # Optimize
        @info "Starting optimization..."
        optimize!(solver.model)
        # Create solution statistics
        stats = SolutionStats(
            solve_time(solver.model),
            solver.setup_time,
            node_count(solver.model),
            objective_value(solver.model),
            objective_bound(solver.model),
            relative_gap(solver.model),
            termination_status(solver.model),
            solver.stats  # Include callback statistics
        )
        println("\nSolution status: ", stats.status)
        
         # Create solution if optimal or time limit or node limit reached
         if termination_status(solver.model) in [MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.NODE_LIMIT]
            selected_edges = findall(value.(solver.x) .> 0.5)
            scenario_values = value.(solver.z) .> 0.5

            solution = PPMPSolution(
                solver.instance,
                selected_edges,
                scenario_values,
                objective_value(solver.model),
                stats
            )
            
            # Validate solution
            if config.callback_print_level >= 1
                # Print detailed statistics
                print_solution_stats(solution)

                validation_result = validate_solution(solution)
                if !validation_result["is_feasible"]
                    @warn "Solution validation failed!"
                end
            end

            return solution
        else
            @warn "Solver terminated with status: $(termination_status(solver.model))"
            return nothing
        end

    catch e
        println("\nError during solve: ", e)
        println(stacktrace())
        return nothing
    end
end



"""
Find violated cuts for current solution using built-in mincut function
Returns a vector of most violated cuts
"""
function find_violated_cuts(solver::FlowCutSolver, 
                          x_val::Vector{Float64}, 
                          z_val::Vector{Float64};
                          max_cuts::Int=1,          # Maximum number of cuts to return
                          max_scenarios::Int=1,     # Maximum number of cuts to obtain
                          min_violation::Float64=1e-6,
                          is_whole_search::Bool=true   # If true, search all scenarios by max_scenarios
                          )
    violated_cuts = Cut[]
    
    # Select scenarios to check - prioritize those with higher z values
    # scenario_priorities = [(k, z_val[k]) for k in 1:length(solver.instance.scenarios)]

    subfixed_scenarios = solver.presolve_info !== nothing ? solver.presolve_info.subfixed_scenarios : Int[]

    scenario_priorities = [(k, z_val[k]) for k in 1:length(solver.instance.scenarios)
                      if k ∉ subfixed_scenarios && z_val[k] >= solver.config.user_scenario_relaxation_value_threshold] # Relaxation threshold to attempt violated generation

    # Sort by decreasing z values
    sort!(scenario_priorities, by=x->x[2], rev=true)
    
    for (k, z_k) in scenario_priorities[1:min(max_scenarios, end)]
        
        # Update network capacities for current solution values
        update_network_capacities!(
            solver.networks[k],
            x_val,
            z_k,
            solver.instance.edge_to_index
        )
        
        network = solver.networks[k]
        
        # Get mincut directly using the weighted digraph
        source_part, sink_part, flow_val = custom_mincut(
            network.graph,
            network.source,
            network.sink
            )
        # If max flow < n*z_k, we have a violated cut
        if flow_val < solver.instance.n * z_k - 1e-6
            if solver.config.callback_print_level >= 2
                println("violated at ", k, " Flow value: ", flow_val, "required: ", solver.instance.n * z_k)
            end
            # Extract M and N from source_part
            M = [i for i in network.left_nodes if i in source_part]
            N = [j-solver.instance.n for j in network.right_nodes if j in sink_part]
            
            # Get cut coefficients
            coefficients, coefficients_norm = generate_cut_coefficients(solver, k, M, N)   
            
            # Calculate violation
            val_vec = vcat(x_val, z_val)
            violation = -sum(coefficients .* val_vec)  # negative since cut is in form ≥ 0
            
            if violation > min_violation
                 # Create Cut object only once with all information
                cut = Cut(coefficients, 0.0, :flow_cut, violation, coefficients_norm, k)
                push!(violated_cuts, cut)
                
                if !is_whole_search
                    if length(violated_cuts) >= max_cuts
                        return violated_cuts
                    end
                end
            end
        else
            if solver.config.callback_print_level >= 2
                println("not violated at ", k, " Flow value: ", flow_val, "required: ", solver.instance.n * z_k)
            end
        end
    end
    
    # Sort cuts by violation from high to low
    # sort!(violated_cuts, by=c -> c.violation, rev=true)
    sort!(violated_cuts, by=c -> c.violation / c.norm, rev=true)
    # Only return the first max_cuts number of cuts
    if length(violated_cuts) > max_cuts
        violated_cuts = violated_cuts[1:max_cuts]
    end

    if !isempty(violated_cuts)
    end
    
    return violated_cuts
end

"""
Generate cut coefficients for given M,N sets
"""
function generate_cut_coefficients(solver::FlowCutSolver, k::Int, M::Vector{Int}, N::Vector{Int})
    # Initialize coefficients for all variables [x; z]
    coefficients = zeros(solver.num_x + solver.num_z)
    
    coefficient_norm = 0.0
    # Set x coefficients (first part of the vector)
    for (i,j) in solver.instance.scenarios[k]
        if i in M && j in N
            edge_idx = solver.instance.edge_to_index[(i,j)]
            coefficients[edge_idx] = 1.0
            coefficient_norm += 1.0
        end
    end
    
    # Set z coefficient (only for scenario k)
    # z coefficient is -(|M| + |N| - n)
    coefficients[solver.num_x + k] = -(length(M) + length(N) - solver.instance.n)
    coefficient_norm += (coefficients[solver.num_x + k])^2
    
    return coefficients, coefficient_norm
end

"""
Custom mincut implementation that accepts SimpleWeightedDiGraph
Following the official mincut implementation structure with adaptation for SimpleWeightedDiGraph
Returns (source_part, sink_part, flow_value) similar to the original GraphsFlows.mincut

Custom mincut implementation optimized for bipartite flow network structure
"""
function custom_mincut(
    graph::SimpleWeightedDiGraph{T,U}, 
    source::Integer,
    sink::Integer,
) where {T,U}
    # n = (nv(graph) - 2) ÷ 2  # number of nodes on each side

    # Get flow value, flow matrix, and labels using B-K algorithm
    flow_val, flow_matrix, labels = maximum_flow(
        graph, 
        source, 
        sink, 
        weights(graph), 
        algorithm=BoykovKolmogorovAlgorithm()
    )

    # Convert labels to partitions
    # Labels from B-K are typically:
    # 0: unlabeled nodes
    # 1: nodes in source tree (S)
    # 2: nodes in sink tree (T)
    
    # Create the two partitions
    source_part = Vector{Int}()
    sink_part = Vector{Int}()
    
    # Iterate through all vertices except source and sink
    for v in vertices(graph)
        if v != source && v != sink
            if labels[v] == 1  # Node is in source tree
                push!(source_part, v)
            else  # Node is either in sink tree (2) or unlabeled (0)
                push!(sink_part, v)
            end
        end
    end
    
    return (source_part, sink_part, flow_val)
end
