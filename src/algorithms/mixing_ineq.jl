# src/algorithms/mixing_ineq.jl

using JuMP
using MinCostFlows

"""
Compute q_k_k'(M,N) using the pre-built cost network structure
Uses MinCostFlows package's in-place updates and warm-starts
"""
function compute_q_kk_prime_MN(solver::FlowCutSolver, k::Int, k_prime::Int, cut::Cut)
    if solver.config.callback_print_level >= 3
        println("compute q_kkprime$(k)$(k_prime)")
    end

    try
        fp = solver.cost_networks[k_prime]
        offset = 2 * solver.instance.n  # Skip s->L and R->t edges
        
        # First reset all L->R arc costs to 0
        for idx in 1:length(solver.instance.scenarios[k_prime])
            updateflowcost!(fp.edges[offset + idx], 0)
        end

        # Update costs for L->R arcs: cost 1 if edge appears in cut, 0 otherwise
        for (idx, (i,j)) in enumerate(solver.instance.scenarios[k_prime])
            edge_idx = solver.instance.edge_to_index[(i,j)]
            updateflowcost!(fp.edges[offset + idx], 
                          abs(cut.coefficients[edge_idx]) > 1e-6 ? 1 : 0)
        end
        
        # Solve using warm start from previous solution
        solveflows!(fp)
        
        # Get total cost
        total_cost = sum(flow * cost for (flow, cost) in zip(flows(fp), costs(fp)))
        
        if solver.config.callback_print_level >= 3
            println("q value: $total_cost")
        end
        
        return total_cost
        
    catch e
        if solver.config.callback_print_level >= 2
            println("Error in compute_q_kk_prime_MN: ", e)
            println(stacktrace())
        end
        rethrow(e)
    end
end

function generate_mixing_coefficients(solver::FlowCutSolver, k::Int, cut::Cut, z_val::Vector{Float64})
    # Skip if k is subfixed
    if solver.presolve_info !== nothing && k in solver.presolve_info.subfixed_scenarios
        throw(ErrorException("Cannot generate mixing cuts for subfixed scenario $k"))
    end
    # Get number of scenarios and scenario probabilities
    # Get scenarios excluding subfixed ones
    
    # Get active scenarios (excluding subfixed ones)
    active_scenarios = if solver.presolve_info !== nothing
        setdiff(1:solver.num_z, solver.presolve_info.subfixed_scenarios)
    else
        1:solver.num_z
    end

    # num_scenarios = solver.num_z
    probabilities = solver.instance.probabilities[active_scenarios]
    
    # Compute q_k_k'(M,N) values for ALL scenarios
    q_values = Float64[]
    k_indices = Int[]
    
    # Compute q values for ALL scenarios (no filtering)
    for k_prime in active_scenarios
        try
            q = compute_q_kk_prime_MN(solver, k, k_prime, cut)
            push!(q_values, q)
            push!(k_indices, k_prime)
        catch e
            if solver.config.callback_print_level >= 2
                println("Error computing q value for scenarios $k, $k_prime: ", e)
            end
            continue
        end
    end
    
    if isempty(q_values)
        throw(ErrorException("No feasible q values found for mixing"))
    end

    # Sort scenarios by decreasing q values while keeping track of indices and probabilities
    sorted_pairs = sort(collect(zip(q_values, k_indices, probabilities)), rev=true)
    sorted_q = [p[1] for p in sorted_pairs]
    sorted_k = [p[2] for p in sorted_pairs]
    sorted_p = [p[3] for p in sorted_pairs]

    # Find smallest index idx such that sum(p[idx:end]) ≥ 1 - epsilon
    epsilon = solver.presolve_info !== nothing ? solver.presolve_info.new_epsilon : solver.instance.epsilon
    
    # This ensures we select enough scenarios to satisfy probability requirement
    cumulative_prob = 0.0
    q_idx = length(sorted_q)
    for i in reverse(1:length(sorted_q))
        cumulative_prob += sorted_p[i]
        if cumulative_prob >= 1 - epsilon - 1e-7
            q_idx = i
            break
        end
    end

    # The constant term will be q[q_idx]
    constant_q = sorted_q[q_idx]
    
    # Generate mixing inequality coefficients
    coefficients = zeros(length(cut.coefficients))
    coefficients[1:solver.num_x] = cut.coefficients[1:solver.num_x]
    


       # Only proceed if we have enough scenarios to generate mixing inequalities
    if q_idx > 1
        if solver.config.mixing_print_level >= 1
            println("q for consider: ", sorted_q[1:q_idx])
            println("z for consider: ", z_val[sorted_k[1:q_idx]])
            println("k for consider: ", sorted_k[1:q_idx])
        end
        # Initialize working arrays
        available_positions = collect(1:(q_idx-1))  # Only consider positions up to q_idx-1
        selected_positions = Int[]
        
        while !isempty(available_positions)
            # Find index with largest z value among available positions
            max_z_val = -Inf
            max_z_pos = 0
            # for pos in reverse(available_positions)
            for pos in available_positions
                if z_val[sorted_k[pos]] > max_z_val
                    max_z_val = z_val[sorted_k[pos]]
                    max_z_pos = pos
                end
            end
            
            if max_z_pos == 0  # No valid position found
                break
            end
            
            # Get the scenario index
            k_prime = sorted_k[max_z_pos]
            
            # Calculate coefficient
            if isempty(selected_positions)
                # First selected index: q[k₁] - q[idx]
                coefficients[solver.num_x + k_prime] = -(sorted_q[max_z_pos] - constant_q)
            else
                # Subsequent indices: q[kᵢ] - q[kᵢ₋₁]
                prev_pos = selected_positions[end]
                coefficients[solver.num_x + k_prime] = -(sorted_q[max_z_pos] - sorted_q[prev_pos])
            end
            
            # Update tracking arrays
            push!(selected_positions, max_z_pos)
            if solver.config.mixing_print_level >= 1
                # println("Selected position: ", max_z_pos)
                println("Selected positions: ", sorted_k[selected_positions])
            end
            # Only keep positions that come before the current position
            filter!(p -> p < max_z_pos, available_positions)
        end
    else
        if solver.config.mixing_print_level >= 1
            println("No scenarios to consider for mixing")
            println("q for 10-th first: ", sorted_q[1:10])
        end
    end


    if solver.config.mixing_print_level >= 2
        println("Required scenarios index: $q_idx")
        println("q values (descending): ", round.(sorted_q[1:min(5, length(sorted_q))], digits=2), 
                length(sorted_q) > 5 ? "..." : "")
        println("Corresponding probabilities: ", round.(sorted_p[1:min(5, length(sorted_p))], digits=4))
        println("constant_q: $constant_q")
    end

    if solver.config.mixing_print_level >= 1
        println("Solver presolve_info: ", solver.presolve_info !== nothing)

        println("\nMixing Details for Scenario $k:")
        println("--------------------------------")
        println("Original epsilon: ", solver.instance.epsilon)
        if solver.presolve_info !== nothing
            println("Updated epsilon (after subfixed): ", solver.presolve_info.new_epsilon)
            println("Number of subfixed scenarios: ", length(solver.presolve_info.subfixed_scenarios))
            println("Subfixed scenarios: ", solver.presolve_info.subfixed_scenarios)
        end
        
        println("\nScenario Processing:")
        println("Active scenarios considered: ", length(active_scenarios))
        println("q values computed: ", length(q_values))
        
        println("\nSorting Results:")
        println("Required q_idx: ", q_idx)
        println("Constant q value: ", constant_q)
        
        println("\nSelected Scenarios for Mixing:")
        for (i, pos) in enumerate(selected_positions)
            k_selected = sorted_k[pos]
            coeff = coefficients[solver.num_x + k_selected]
            println("Position $i: Scenario $(k_selected)")
            println("  q value: $(sorted_q[pos])")
            println("  z value: $(z_val[k_selected])")
            println("  probability: $(solver.instance.probabilities[k_selected])")
            println("  coefficient in mix: $coeff")
        end
        
        println("\nFinal Mixing Inequality:")
        println("Number of non-zero x coefficients: ", count(!iszero, coefficients[1:solver.num_x]))
        println("Number of non-zero z coefficients: ", count(!iszero, coefficients[solver.num_x+1:end]))
        println("Constant term: ", constant_q)
    end


    return coefficients, sorted_q, constant_q
end

"""Obsolete mixing coefficient generation function"""
# function generate_mixing_coefficients(solver::FlowCutSolver, k::Int, cut::Cut, z_val::Vector{Float64})
#     # Get number of scenarios and scenario probabilities
#     num_scenarios = solver.num_z
#     probabilities = solver.instance.probabilities
    
#     # Compute q_k_k'(M,N) values for ALL scenarios
#     q_values = Float64[]
#     k_indices = Int[]
    
#     # Compute q values for ALL scenarios (no filtering)
#     for k_prime in 1:num_scenarios
#         try
#             q = compute_q_kk_prime_MN(solver, k, k_prime, cut)
#             push!(q_values, q)
#             push!(k_indices, k_prime)
#         catch e
#             if solver.config.callback_print_level >= 2
#                 println("Error computing q value for scenarios $k, $k_prime: ", e)
#             end
#             continue
#         end
#     end
    
#     if isempty(q_values)
#         throw(ErrorException("No feasible q values found for mixing"))
#     end

#     # Sort scenarios by decreasing q values while keeping track of indices  
#     sorted_pairs = sort(collect(zip(q_values, k_indices, probabilities)), rev=true)
#     sorted_q = [p[1] for p in sorted_pairs]
#     sorted_k = [p[2] for p in sorted_pairs]
#     sorted_p = [p[3] for p in sorted_pairs]

#     # Find smallest index idx such that sum(p[idx:end]) ≥ 1 - epsilon
#     # This ensures we select enough scenarios to satisfy probability requirement
#     cumulative_prob = 0.0
#     q_idx = length(sorted_q)
#     for i in reverse(1:length(sorted_q))
#         cumulative_prob += sorted_p[i]
#         if cumulative_prob >= 1 - solver.instance.epsilon - 1e-7
#             q_idx = i
#             break
#         end
#     end

#     constant_q = sorted_q[q_idx]
    
#     # Generate mixing inequality coefficients
#     coefficients = zeros(length(cut.coefficients))
#     coefficients[1:solver.num_x] = cut.coefficients[1:solver.num_x]
    
#     # Only proceed if we can generate meaningful mixing inequalities
#     if q_idx > 1
#         println("q for consider: ", sorted_q[1:q_idx])
#         println("k for consider: ", sorted_k[1:q_idx])
#         # Select indices with largest z values while maintaining increasing positions
#         selected_indices = Int[]
#         remaining_positions = collect(1:(q_idx-1))

#         # Adjust q values for mixing by subtracting constant_q
#         # sorted_q[1:q_idx-1] .-= constant_q
        
#         while !isempty(remaining_positions)
#             valid_positions = remaining_positions
#             if !isempty(selected_indices)
#                 valid_positions = filter(p -> p > selected_indices[end], remaining_positions)
#             end
            
#             if isempty(valid_positions)
#                 break
#             end
            
#             # Select position with largest z value
#             next_pos = valid_positions[argmax(i -> z_val[sorted_k[valid_positions[i]]], 1:length(valid_positions))]
#             push!(selected_indices, next_pos)
            
#             # Update remaining positions
#             remaining_positions = remaining_positions[remaining_positions .> next_pos]
#         end

#         # Add coefficients for selected indices
#         if !isempty(selected_indices)
#             # Handle first coefficient: -(max_q - q_i1)
#             first_pos = selected_indices[1]
#             coefficients[solver.num_x + sorted_k[first_pos]] = -(sorted_q[1] - sorted_q[first_pos])

#             # Handle middle coefficients: -(q_ik-1 - q_ik)
#             for i in 1:length(selected_indices)-1
#                 curr_pos = selected_indices[i]
#                 next_pos = selected_indices[i+1]
#                 coefficients[solver.num_x + sorted_k[curr_pos]] = -(sorted_q[curr_pos] - sorted_q[next_pos])
#             end

#             # Handle last coefficient: -q_ik
#             last_pos = selected_indices[end]
#             coefficients[solver.num_x + sorted_k[last_pos]] = -sorted_q[last_pos]
#         end
#     end

#     if solver.config.mixing_print_level >= 2
#         println("Required scenarios index: $q_idx")
#         println("q values (descending): ", round.(sorted_q[1:min(5, length(sorted_q))], digits=2), 
#                 length(sorted_q) > 5 ? "..." : "")
#         println("Corresponding probabilities: ", round.(sorted_p[1:min(5, length(sorted_p))], digits=4))
#         println("constant_q: $constant_q")
#     end

#     return coefficients, sorted_q, constant_q
# end