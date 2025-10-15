# src/algorithms/mixing_ineq.jl

using JuMP
using MinCostFlows

"""
Check if all scenario probabilities are equal within tolerance.
Used to determine if complement mixing inequalities can be applied.
"""
function has_equal_probabilities(probabilities::Vector{Float64}, tol::Float64=1e-9)::Bool
    if isempty(probabilities)
        return true
    end

    expected_prob = 1.0 / length(probabilities)
    return all(abs(p - expected_prob) < tol for p in probabilities)
end

"""
Compute Delta coefficients for complement variables using recursive formula.
This implements Definition 2 from the theoretical framework.

Arguments:
- sorted_q: q (or h) values sorted in DECREASING order [q[1] >= q[2] >= ... >= q[n]]
- m: split parameter (must satisfy 1 <= m < length(sorted_q))
- num_complements: number of Delta coefficients to compute (must be > 0)

Returns:
- Vector{Float64}: [Δ₁, Δ₂, ..., Δ_{num_complements}]

Formula:
- Δ₁ = q[m+1] - q[m+2]
- Δᵢ = max(Δᵢ₋₁, q[m+1] - q[m+i+1] - Σⱼ₌₁ⁱ⁻¹ Δⱼ) for i >= 2
"""
function compute_delta_coefficients(sorted_q::Vector{Float64}, m::Int, num_complements::Int)::Vector{Float64}
    if num_complements <= 0
        return Float64[]
    end

    # Verify indices are valid
    max_index_needed = m + num_complements + 1
    if max_index_needed > length(sorted_q)
        error("Cannot compute $num_complements Delta coefficients: " *
              "need sorted_q[$max_index_needed] but only have $(length(sorted_q)) elements")
    end

    deltas = zeros(Float64, num_complements)

    # Base case
    deltas[1] = sorted_q[m+1] - sorted_q[m+2]

    # Recursive cases
    for i in 2:num_complements
        option1 = deltas[i-1]
        option2 = sorted_q[m+1] - sorted_q[m+i+1] - sum(deltas[1:i-1])
        deltas[i] = max(option1, option2)
    end

    return deltas
end

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

"""
Generate complement mixing inequality coefficients (Theorem 4 type).
This is DIFFERENT from basic star inequalities - uses complement variables.

Key differences from generate_mixing_coefficients:
1. Selects T from positions {1, ..., m} instead of {1, ..., p}
2. Also selects complement scenarios Q from positions {p+1, ..., n}
3. Uses different constant term: h[m+1] - sum(Deltas)
4. Only valid for EQUAL probabilities

Arguments: (same as generate_mixing_coefficients)
- solver: FlowCutSolver instance
- k: scenario index for which cut was generated
- cut: the Cut object
- z_val: current z variable values from LP relaxation

Returns: (same format as generate_mixing_coefficients)
- coefficients: Vector{Float64} of length |E| + |Ω|
- sorted_q: sorted q values (for debugging)
- constant_q: RHS constant term
"""
function generate_complement_mixing_inequality(
    solver::FlowCutSolver,
    k::Int,
    cut::Cut,
    z_val::Vector{Float64}
)
    # Skip if k is subfixed
    if solver.presolve_info !== nothing && k in solver.presolve_info.subfixed_scenarios
        throw(ErrorException("Cannot generate mixing cuts for subfixed scenario $k"))
    end

    # Get active scenarios
    active_scenarios = if solver.presolve_info !== nothing
        setdiff(1:solver.num_z, solver.presolve_info.subfixed_scenarios)
    else
        1:solver.num_z
    end

    probabilities = solver.instance.probabilities[active_scenarios]

    # Check if probabilities are equal - REQUIRED for this method
    # Note: This check should have been done at solver initialization
    # This is a safety check in case function is called incorrectly
    if !has_equal_probabilities(probabilities)
        throw(ErrorException("Complement mixing inequalities require equal probabilities. " *
                            "This should have been checked at solver initialization."))
    end

    # Compute q values for ALL scenarios (same as basic star)
    q_values = Float64[]
    k_indices = Int[]

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

    # Sort by DECREASING q values
    sorted_pairs = sort(collect(zip(q_values, k_indices, probabilities)), rev=true)
    sorted_q = [p[1] for p in sorted_pairs]
    sorted_k = [p[2] for p in sorted_pairs]
    sorted_p = [p[3] for p in sorted_pairs]

    # Find threshold p (same as basic star)
    epsilon = solver.presolve_info !== nothing ?
              solver.presolve_info.new_epsilon :
              solver.instance.epsilon

    cumulative_prob = 0.0
    q_idx = length(sorted_q)
    for i in reverse(1:length(sorted_q))
        cumulative_prob += sorted_p[i]
        if cumulative_prob >= 1 - epsilon - 1e-7
            q_idx = i
            break
        end
    end

    p = q_idx - 1  # Threshold: positions 1 to p are "at risk"

    # Choose m parameter
    # Strategy: Use m slightly less than p to allow room for complements
    # For example: m = max(1, p - 1) ensures at least 1 complement scenario
    m = max(1, p - 1)

    if solver.config.mixing_print_level >= 1
        println("\n=== Complement Mixing Inequality Generation ===")
        println("p (threshold): $p, q_idx: $q_idx")
        println("m (split parameter): $m")
        println("Max positions for T: {1, ..., $m}")
        println("Max complement scenarios: $(p - m)")
    end

    # Initialize coefficients
    coefficients = zeros(length(cut.coefficients))
    coefficients[1:solver.num_x] = cut.coefficients[1:solver.num_x]

    # STEP 1: Select T ⊆ {1, ..., m} based on z values (greedy)
    available_T = collect(1:m)
    selected_T = Int[]

    while !isempty(available_T)
        max_z_val = -Inf
        max_z_pos = 0

        for pos in available_T
            if z_val[sorted_k[pos]] > max_z_val
                max_z_val = z_val[sorted_k[pos]]
                max_z_pos = pos
            end
        end

        if max_z_pos == 0
            break
        end

        push!(selected_T, max_z_pos)

        if solver.config.mixing_print_level >= 2
            println("Selected T position: $max_z_pos (scenario $(sorted_k[max_z_pos])), z=$(round(max_z_val, digits=4))")
        end

        # Keep only positions AFTER current (maintain increasing order)
        filter!(p -> p > max_z_pos, available_T)
    end

    # STEP 2: Add coefficients for T set
    # Coefficient formula: -(q[tᵢ] - q[tᵢ₊₁]) where q[t_{|T|+1}] = q[m+1]
    for (idx, pos) in enumerate(selected_T)
        k_prime = sorted_k[pos]

        if idx < length(selected_T)
            next_pos = selected_T[idx + 1]
            coefficients[solver.num_x + k_prime] = -(sorted_q[pos] - sorted_q[next_pos])
        else
            # Last position in T pairs with m+1
            coefficients[solver.num_x + k_prime] = -(sorted_q[pos] - sorted_q[m+1])
        end

        if solver.config.mixing_print_level >= 2
            println("T coefficient for scenario $k_prime: ", coefficients[solver.num_x + k_prime])
        end
    end

    # STEP 3: Select Q ⊆ {p+1, ..., n} for complement variables
    max_complement_count = p - m
    available_Q = collect(q_idx:length(sorted_q))  # positions p+1 to n
    selected_Q = Int[]

    # Select based on LARGE z values (satisfied scenarios tighten the cut via complements)
    while !isempty(available_Q) && length(selected_Q) < max_complement_count
        max_z_val = -Inf
        max_z_pos = 0

        for pos in available_Q
            if z_val[sorted_k[pos]] > max_z_val
                max_z_val = z_val[sorted_k[pos]]
                max_z_pos = pos
            end
        end

        if max_z_pos == 0
            break
        end

        push!(selected_Q, max_z_pos)
        filter!(p -> p != max_z_pos, available_Q)

        if solver.config.mixing_print_level >= 2
            println("Selected Q position: $max_z_pos (scenario $(sorted_k[max_z_pos])), z=$(round(max_z_val, digits=4))")
        end
    end

    # STEP 4: Compute Delta coefficients and add complement terms
    constant_q = sorted_q[m+1]  # Base constant

    if !isempty(selected_Q)
        try
            deltas = compute_delta_coefficients(sorted_q, m, length(selected_Q))

            # Add complement coefficients
            # In 2010 paper: uses tilde_z (0=satisfied, 1=violated)
            # In our code: uses z (1=satisfied, 0=violated)
            # Paper has: + Δ * tilde_z = + Δ * (1-z) = Δ - Δ*z
            # So our coefficient on z is: -Δ
            for (idx, q_pos) in enumerate(selected_Q)
                k_prime = sorted_k[q_pos]
                coefficients[solver.num_x + k_prime] = -deltas[idx]

                if solver.config.mixing_print_level >= 2
                    println("Q coefficient for scenario $k_prime: -Δ=$(round(-deltas[idx], digits=4))")
                end
            end

            # Adjust constant: subtract sum of Deltas
            # Paper: RHS = h[m+1] - Σ Δ (because of the +Δ*tilde_z terms)
            delta_sum = sum(deltas)
            constant_q -= delta_sum

            if solver.config.mixing_print_level >= 1
                println("Added $(length(selected_Q)) complement variables")
                println("Delta values: ", round.(deltas, digits=4))
                println("Constant adjusted from $(constant_q + delta_sum) to $constant_q")
            end

        catch e
            if solver.config.mixing_print_level >= 1
                println("Warning: Failed to compute Delta coefficients: ", e)
                println("Falling back to basic inequality without complements")
            end
            # Keep the T coefficients, but no complement adjustment
        end
    else
        if solver.config.mixing_print_level >= 1
            println("No complement scenarios selected (m=$m, p=$p)")
        end
    end

    if solver.config.mixing_print_level >= 1
        println("Final mixing inequality:")
        println("  |T| = $(length(selected_T))")
        println("  |Q| = $(length(selected_Q))")
        println("  Constant: $constant_q")
        println("==============================================\n")
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