# src/algorithms/original_mip.jl

using JuMP
using CPLEX
using PPMP

"""
Solver using the original MIP flow formulation
"""
struct OriginalMIPSolver <: PPMPSolver
    model::Model
    instance::PPMPInstance
    x::Vector{VariableRef}    # Edge selection variables
    z::Vector{VariableRef}    # Scenario selection variables
    y::Vector{Dict{Tuple{Int,Int}, VariableRef}}  # Flow variables for each scenario
    setup_time::Float64
    config::PPMPConfig
end


"""
Initialize original MIP solver
"""
function OriginalMIPSolver(instance::PPMPInstance; config::PPMPConfig=default_config())
    start_time = time()

    @info "Creating CPLEX optimizer..."
    model = Model(CPLEX.Optimizer)
    @info "CPLEX optimizer Successfully created"
    # set_silent(model)
    
    # Create edge selection variables with names
    x = @variable(model, [i=1:length(instance.edges)], Bin, base_name="x")
    
    # Create scenario selection variables with names
    z = @variable(model, [i=1:length(instance.scenarios)], Bin, base_name="z")
    
    # Create flow variables for each scenario with meaningful names
    y = Vector{Dict{Tuple{Int,Int}, VariableRef}}(undef, length(instance.scenarios))
    
    n = instance.n
    s, t = 2n + 1, 2n + 2

    # For each scenario, create its flow network and variables
    for k in 1:length(instance.scenarios)
        scenario_edges = instance.scenarios[k]
        flow_arcs = create_flow_network(instance.n, scenario_edges)
        y[k] = Dict{Tuple{Int,Int}, VariableRef}()
        
        for arc in flow_arcs
            # Create descriptive names for flow variables
            if arc == (t, s)  # t->s arc (corrected naming)
                name = "y$(k)_t_s"
            elseif arc[1] == s  # s->L arcs
                name = "y$(k)_s_$(arc[2])"
            elseif arc[2] == t  # R->t arcs
                name = "y$(k)_$(arc[1]-n)_t"
            else  # L->R arcs
                name = "y$(k)_$(arc[1])_$(arc[2]-n)"
            end
            y[k][arc] = @variable(model, lower_bound=0, base_name=name)
        end
        
        # Add flow conservation constraints with names
        n = instance.n
        s, t = 2n + 1, 2n + 2
        
        for u in vcat(1:2n, [s, t])
            # Find incoming and outgoing arcs
            incoming = filter(a -> a[2] == u, flow_arcs)
            outgoing = filter(a -> a[1] == u, flow_arcs)
            
            # Add named flow conservation constraint
            node_type = u == s ? "source" : (u == t ? "sink" : (u ≤ n ? "L$u" : "R$(u-n)"))
            @constraint(model, 
                sum(y[k][arc] for arc in incoming) == 
                sum(y[k][arc] for arc in outgoing),
                base_name="flow_cons_$(k)_$(node_type)")
        end
        
        # Add capacity constraints with names
        for (i,j) in scenario_edges
            edge_idx = instance.edge_to_index[(i,j)]
            @constraint(model, y[k][(i,n+j)] <= x[edge_idx],
                       base_name="cap_cons_$(k)_$(i)_$(j)")
        end
        
        # Add s->L and R->t capacity constraints with names
        for i in 1:n
            @constraint(model, y[k][(s,i)] <= z[k],
                       base_name="source_cap_$(k)_$(i)")
            @constraint(model, y[k][(n+i,t)] <= z[k],
                       base_name="sink_cap_$(k)_$(i)")
        end
        
        # Add flow requirement constraint with name
        @constraint(model, y[k][(t,s)] >= instance.n * z[k],
                   base_name="flow_req_$(k)")
    end
    
    # Add objective with name
    @objective(model, Min, sum(instance.costs .* x))
    
    # Add chance constraint with name
    @constraint(model, sum(instance.probabilities .* z) >= 1 - instance.epsilon,
               base_name="chance_cons")
    
    setup_time = time() - start_time
    return OriginalMIPSolver(model, instance, x, z, y, setup_time, config)
end

"""
Solve PPMP instance using original MIP formulation
"""
function solve(solver::OriginalMIPSolver)
    config = solver.config
    try
        # Set some CPLEX parameters
        set_optimizer_attribute(solver.model, "CPX_PARAM_EPGAP", config.mip_gap)  # Relative gap tolerance
        set_optimizer_attribute(solver.model, "CPX_PARAM_TILIM", config.max_time)  # Solving time limit
        set_optimizer_attribute(solver.model, "CPX_PARAM_THREADS", config.threads)   # Threads Number
        
        # set_optimizer_attribute(solver.model, "CPX_PARAM_PREIND", 0)  # Disable presolve for callbacks

        """Auto Benders setup"""
        if config.auto_benders
            # Enable CPLEX auto-benders with full decomposition strategy. It automatically recognizes the integer variables and put them in the master problem.
            set_optimizer_attribute(solver.model, "CPXPARAM_Benders_Strategy", 3)
        end

        # Optimize
        @info "Starting optimization..."
        optimize!(solver.model)
        
        # Check solution status
        status = termination_status(solver.model)
        println("\nSolution status: ", status)

        if status == MOI.INFEASIBLE
            println("\nModel is infeasible. Computing IIS...")
            # Note: CPLEX specific command to compute Irreducible Infeasible Subsystem
            set_optimizer_attribute(solver.model, "CPX_PARAM_IISFIND", 1)
            optimize!(solver.model)
            return nothing
        elseif status != MOI.OPTIMAL && status != MOI.TIME_LIMIT
            println("\nUnexpected status: ", status)
            return nothing
        end
        
        # Collect solution statistics
        stats = SolutionStats(
            solve_time(solver.model),  # Actual solve time
            solver.setup_time,                # set up time
            node_count(solver.model),
            objective_value(solver.model),
            objective_bound(solver.model),
            relative_gap(solver.model),
            termination_status(solver.model)
        )

        # Get solution
        selected_edges = findall(value.(solver.x) .> 0.5)
        scenario_values = value.(solver.z) .> 0.5
        
        return PPMPSolution(
            solver.instance,
            selected_edges,
            scenario_values,
            objective_value(solver.model),
            stats
        )
    catch e
        println("\nError during solve: ", e)
        return nothing
    end
end