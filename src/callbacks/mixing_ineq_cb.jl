# src/callbacks/mixing_ineq_cb.jl

"""
Check if a mixing inequality is trivial, i.e.:
1. Same as original cut (same z coefficients and zero constant term)
2. Only x variables with zero constant term
"""
function is_trivial_mixing_inequality(mixing_coeffs::Vector{Float64}, 
                                    original_coeffs::Vector{Float64}, 
                                    constant_q::Float64,
                                    num_x::Int,
                                    k::Int)
    # Check if constant term is effectively zero
    if abs(constant_q) < 1e-6
        # Case 1: Same as original cut
        if abs(mixing_coeffs[num_x + k] - original_coeffs[num_x + k]) < 1e-6 &&
           all(abs(coeff) < 1e-6 for (i, coeff) in enumerate(mixing_coeffs[(num_x+1):end]) if i != k)
            return true
        end
        
        # Case 2: Only x variables (all z coefficients are zero)
        if all(abs(coeff) < 1e-6 for coeff in mixing_coeffs[(num_x+1):end])
            return true
        end
    end
    
    return false
end

"""
Print mixing inequality in readable format
"""
function print_mixing_inequality(num_x::Int, mixing_coeffs::Vector{Float64}, 
                               constant_q::Float64, k::Int, cut_type::String="")
    z_terms = String[]
    for i in (num_x + 1):length(mixing_coeffs)
        if abs(mixing_coeffs[i]) > 1e-6
            scenario_idx = i - num_x
            push!(z_terms, @sprintf("%.2f z_%d", mixing_coeffs[i], scenario_idx))
        end
    end
    z_part = isempty(z_terms) ? "" : " + " * join(z_terms, " + ")
    
    # prefix = isempty(cut_type) ? "" : "\nProcessing $cut_type mixing cut for scenario $k:"
    # if !isempty(prefix)
    #     println(prefix)
    # end
    println("Mixing inequality for scenario $k: x(M,N)$z_part ≥ $constant_q")
end

"""
Process mixing cuts for lazy constraints
"""
function process_mixing_cuts_lazy!(cb_data, solver::FlowCutSolver, config::PPMPConfig,
                                 cuts::Vector{Cut}, x_val::Vector{Float64}, z_val::Vector{Float64}, is_root::Bool)

    max_mixing_cuts_per_round = is_root ? config.root_max_lazy_mixing_cuts_per_round : config.tree_max_lazy_mixing_cuts_per_round
    max_mixing_cuts_submit_per_round = is_root ? config.root_max_lazy_mixing_cuts_submit_per_round : config.tree_max_lazy_mixing_cuts_submit_per_round

    if !config.lazy_mixing_cuts_enabled || isempty(cuts) || (max_mixing_cuts_per_round == 0) || (max_mixing_cuts_submit_per_round == 0)
        return 0
    end

    lazy_mixing_cuts_added = 0
    lazy_mixing_start_time = time()
    
    if config.mixing_print_level >= 2
        println("Processing mixing cut for most violated cut")
    end

    # Store candidate cuts
    candidate_cuts = []
    
    for cut_idx in 1:min(length(cuts), max_mixing_cuts_per_round)
        try
            # Find which scenario this cut is for
            k = cuts[cut_idx].k_idx

            # Generate appropriate mixing inequality(ies)
            candidate_inequalities = []
            can_use_complement = solver.has_equal_probs

            # Generate basic_star if enabled
            if config.mixing_basic_star_enabled
                try
                    mixing_coeffs_basic, q_values_basic, constant_q_basic = generate_mixing_coefficients(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_basic, cuts[cut_idx].coefficients,
                                                     constant_q_basic, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_basic = -sum(mixing_coeffs_basic .* val_vec) + constant_q_basic
                        coeff_norm_basic = sqrt(sum(mixing_coeffs_basic.^2))

                        push!(candidate_inequalities, (
                            type = :basic_star,
                            coefficients = mixing_coeffs_basic,
                            constant = constant_q_basic,
                            violation = violation_basic,
                            norm = coeff_norm_basic,
                            normalized_violation = violation_basic / coeff_norm_basic
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated basic_star inequality, violation: $(round(violation_basic, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate basic_star inequality: ", e)
                    end
                end
            end

            # Generate complement if enabled and possible
            if config.mixing_complement_enabled && can_use_complement
                try
                    mixing_coeffs_comp, q_values_comp, constant_q_comp = generate_complement_mixing_inequality(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_comp, cuts[cut_idx].coefficients,
                                                     constant_q_comp, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_comp = -sum(mixing_coeffs_comp .* val_vec) + constant_q_comp
                        coeff_norm_comp = sqrt(sum(mixing_coeffs_comp.^2))

                        push!(candidate_inequalities, (
                            type = :complement,
                            coefficients = mixing_coeffs_comp,
                            constant = constant_q_comp,
                            violation = violation_comp,
                            norm = coeff_norm_comp,
                            normalized_violation = violation_comp / coeff_norm_comp
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated complement inequality, violation: $(round(violation_comp, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate complement inequality: ", e)
                    end
                end
            end

            # Generate improved if enabled
            if config.mixing_improved_enabled
                try
                    mixing_coeffs_improv, q_values_improv, constant_q_improv = generate_improved_mixing_coefficients(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_improv, cuts[cut_idx].coefficients,
                                                     constant_q_improv, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_improv = -sum(mixing_coeffs_improv .* val_vec) + constant_q_improv
                        coeff_norm_improv = sqrt(sum(mixing_coeffs_improv.^2))

                        push!(candidate_inequalities, (
                            type = :improved,
                            coefficients = mixing_coeffs_improv,
                            constant = constant_q_improv,
                            violation = violation_improv,
                            norm = coeff_norm_improv,
                            normalized_violation = violation_improv / coeff_norm_improv
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated improved inequality, violation: $(round(violation_improv, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate improved inequality: ", e)
                    end
                end
            end

            # Generate improved_complement if enabled and possible
            if config.mixing_improved_complement_enabled && can_use_complement
                try
                    mixing_coeffs_improv_comp, q_values_improv_comp, constant_q_improv_comp = generate_improved_complement_mixing_coefficients(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_improv_comp, cuts[cut_idx].coefficients,
                                                     constant_q_improv_comp, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_improv_comp = -sum(mixing_coeffs_improv_comp .* val_vec) + constant_q_improv_comp
                        coeff_norm_improv_comp = sqrt(sum(mixing_coeffs_improv_comp.^2))

                        push!(candidate_inequalities, (
                            type = :improved_complement,
                            coefficients = mixing_coeffs_improv_comp,
                            constant = constant_q_improv_comp,
                            violation = violation_improv_comp,
                            norm = coeff_norm_improv_comp,
                            normalized_violation = violation_improv_comp / coeff_norm_improv_comp
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated improved_complement inequality, violation: $(round(violation_improv_comp, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate improved_complement inequality: ", e)
                    end
                end
            end

            # Select best inequality based on violation
            if !isempty(candidate_inequalities)
                # Sort by normalized violation
                sort!(candidate_inequalities, by=x -> x.normalized_violation, rev=true)

                # Take the best one
                best_inequality = candidate_inequalities[1]

                # Store cut information
                push!(candidate_cuts, (
                    k_idx=k,
                    type=best_inequality.type,
                    coefficients=best_inequality.coefficients,
                    constant=best_inequality.constant,
                    violation=best_inequality.violation,
                    norm=best_inequality.norm,
                    normalized_violation=best_inequality.normalized_violation
                ))

                if config.mixing_print_level >= 1
                    println("Selected $(best_inequality.type) inequality (best violation)")
                    print_mixing_inequality(solver.num_x, best_inequality.coefficients, best_inequality.constant, k, "lazy")
                    println("Violation: ", round(best_inequality.violation, digits=6))
                    println("Normalized violation: ", round(best_inequality.normalized_violation, digits=6))
                end
            end

        catch e
            if config.mixing_print_level >= 1
                println("Failed to generate lazy mixing cut: ", e)
                println(stacktrace())
            end
            continue
        end
    end

    if !isempty(candidate_cuts)
        # filter out cuts with normalized violation below threshold
        candidate_cuts = filter(x -> x.violation > config.lazy_mixing_cuts_violation_threshold, candidate_cuts)
        # Sort candidate cuts by normalized violation
        sort!(candidate_cuts, by=x -> x.normalized_violation, rev=true)
        
        for _ in 1:max_mixing_cuts_submit_per_round
            
            if isempty(candidate_cuts)
                break
            end

            best_cut = pop!(candidate_cuts)
            
            # Add mixing cut with proper RHS
            vars = vcat(solver.x, solver.z)
            mixing_con = @build_constraint(
                sum(best_cut.coefficients[i] * vars[i] for i in 1:length(vars)) >= best_cut.constant
            ) 
            MOI.submit(solver.model, MOI.LazyConstraint(cb_data), mixing_con)
            
            if config.mixing_print_level >= 1
                print_mixing_inequality(solver.num_x, best_cut.coefficients, best_cut.constant, best_cut.k_idx, "lazy")
                println("Violation: ", round(best_cut.violation, digits=6))
                println("Normalized violation: ", round(best_cut.normalized_violation, digits=6))
            end
            
            if config.mixing_print_level >= 3
                println("Adding lazy mixing cut:", mixing_con)
            end
            
            # Update statistics
            lazy_mixing_cuts_added += 1
            solver.stats.lazy_mixing_cuts_added += 1
            if is_root
                solver.stats.root_stats["lazy_mixing_cuts_added"] += 1
            else
                solver.stats.tree_stats["lazy_mixing_cuts_added"] += 1
            end
            
        end
    end

    println("submitted ", lazy_mixing_cuts_added, " lazy mixing cuts. Total: ", solver.stats.root_stats["lazy_mixing_cuts_added"] + solver.stats.tree_stats["lazy_mixing_cuts_added"])

    # Update timing statistics
    solver.stats.lazy_mixing_cuts_time += time() - lazy_mixing_start_time
    
   
    return lazy_mixing_cuts_added
end

"""
Process mixing cuts for uesr cuts
"""
function process_mixing_cuts_user!(cb_data, solver::FlowCutSolver, config::PPMPConfig,
                                 cuts::Vector{Cut}, x_val::Vector{Float64}, z_val::Vector{Float64}, is_root::Bool)

    max_mixing_cuts_per_round = is_root ? config.root_max_user_mixing_cuts_per_round : config.tree_max_user_mixing_cuts_per_round

    max_mixing_cuts_submit_per_round = is_root ? config.root_max_user_mixing_cuts_submit_per_round : config.tree_max_user_mixing_cuts_submit_per_round

    # Track which scenarios have mixing cuts submitted
    submitted_scenario_indices = Set{Int}()

    if !config.user_mixing_cuts_enabled || isempty(cuts) || (max_mixing_cuts_per_round == 0) || (max_mixing_cuts_submit_per_round == 0)
        return 0, submitted_scenario_indices
    end
    

    user_mixing_cuts_added = 0
    user_mixing_start_time = time()
    
    if config.mixing_print_level >= 2
        println("Processing user mixing cut for most violated cut")
    end

    # Store all candidate mixing cuts
    candidate_cuts = []
    
    for cut_idx in 1:min(length(cuts), max_mixing_cuts_per_round)
        try
            # Find which scenario this cut is for
            k = cuts[cut_idx].k_idx

            # Generate appropriate mixing inequality(ies)
            candidate_inequalities = []
            can_use_complement = solver.has_equal_probs

            # Generate basic_star if enabled
            if config.mixing_basic_star_enabled
                try
                    mixing_coeffs_basic, q_values_basic, constant_q_basic = generate_mixing_coefficients(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_basic, cuts[cut_idx].coefficients,
                                                     constant_q_basic, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_basic = -sum(mixing_coeffs_basic .* val_vec) + constant_q_basic
                        coeff_norm_basic = sqrt(sum(mixing_coeffs_basic.^2))

                        push!(candidate_inequalities, (
                            type = :basic_star,
                            coefficients = mixing_coeffs_basic,
                            constant = constant_q_basic,
                            violation = violation_basic,
                            norm = coeff_norm_basic,
                            normalized_violation = violation_basic / coeff_norm_basic
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated basic_star inequality, violation: $(round(violation_basic, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate basic_star inequality: ", e)
                    end
                end
            end

            # Generate complement if enabled and possible
            if config.mixing_complement_enabled && can_use_complement
                try
                    mixing_coeffs_comp, q_values_comp, constant_q_comp = generate_complement_mixing_inequality(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_comp, cuts[cut_idx].coefficients,
                                                     constant_q_comp, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_comp = -sum(mixing_coeffs_comp .* val_vec) + constant_q_comp
                        coeff_norm_comp = sqrt(sum(mixing_coeffs_comp.^2))

                        push!(candidate_inequalities, (
                            type = :complement,
                            coefficients = mixing_coeffs_comp,
                            constant = constant_q_comp,
                            violation = violation_comp,
                            norm = coeff_norm_comp,
                            normalized_violation = violation_comp / coeff_norm_comp
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated complement inequality, violation: $(round(violation_comp, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate complement inequality: ", e)
                    end
                end
            end

            # Generate improved if enabled
            if config.mixing_improved_enabled
                try
                    mixing_coeffs_improv, q_values_improv, constant_q_improv = generate_improved_mixing_coefficients(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_improv, cuts[cut_idx].coefficients,
                                                     constant_q_improv, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_improv = -sum(mixing_coeffs_improv .* val_vec) + constant_q_improv
                        coeff_norm_improv = sqrt(sum(mixing_coeffs_improv.^2))

                        push!(candidate_inequalities, (
                            type = :improved,
                            coefficients = mixing_coeffs_improv,
                            constant = constant_q_improv,
                            violation = violation_improv,
                            norm = coeff_norm_improv,
                            normalized_violation = violation_improv / coeff_norm_improv
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated improved inequality, violation: $(round(violation_improv, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate improved inequality: ", e)
                    end
                end
            end

            # Generate improved_complement if enabled and possible
            if config.mixing_improved_complement_enabled && can_use_complement
                try
                    mixing_coeffs_improv_comp, q_values_improv_comp, constant_q_improv_comp = generate_improved_complement_mixing_coefficients(
                        solver, k, cuts[cut_idx], z_val
                    )

                    if !is_trivial_mixing_inequality(mixing_coeffs_improv_comp, cuts[cut_idx].coefficients,
                                                     constant_q_improv_comp, solver.num_x, k)
                        val_vec = vcat(x_val, z_val)
                        violation_improv_comp = -sum(mixing_coeffs_improv_comp .* val_vec) + constant_q_improv_comp
                        coeff_norm_improv_comp = sqrt(sum(mixing_coeffs_improv_comp.^2))

                        push!(candidate_inequalities, (
                            type = :improved_complement,
                            coefficients = mixing_coeffs_improv_comp,
                            constant = constant_q_improv_comp,
                            violation = violation_improv_comp,
                            norm = coeff_norm_improv_comp,
                            normalized_violation = violation_improv_comp / coeff_norm_improv_comp
                        ))

                        if config.mixing_print_level >= 1
                            println("Generated improved_complement inequality, violation: $(round(violation_improv_comp, digits=6))")
                        end
                    end
                catch e
                    if config.mixing_print_level >= 1
                        println("Failed to generate improved_complement inequality: ", e)
                    end
                end
            end

            # Select best inequality based on violation
            if !isempty(candidate_inequalities)
                # Sort by normalized violation
                sort!(candidate_inequalities, by=x -> x.normalized_violation, rev=true)

                # Take the best one
                best_inequality = candidate_inequalities[1]

                # Store cut information
                push!(candidate_cuts, (
                    k_idx=k,
                    type=best_inequality.type,
                    coefficients=best_inequality.coefficients,
                    constant=best_inequality.constant,
                    violation=best_inequality.violation,
                    norm=best_inequality.norm,
                    normalized_violation=best_inequality.normalized_violation
                ))

                if config.mixing_print_level >= 1
                    println("Selected $(best_inequality.type) inequality (best violation)")
                    println("Non-trivial mixing inequality for scenario $k")
                    print_mixing_inequality(solver.num_x, best_inequality.coefficients, best_inequality.constant, k, "user")
                    println("Violation: ", round(best_inequality.violation, digits=6))
                    println("Normalized violation: ", round(best_inequality.normalized_violation, digits=6))
                end
            end

        catch e
            if config.mixing_print_level >= 1
                println("Failed to generate user mixing cut: ", e)
                println(stacktrace())
            end
            continue
        end
    end

    # Sort candidate cuts by normalized violation
    if !isempty(candidate_cuts)
        # candidate_cuts = filter(x -> x.violation > config.user_mixing_cuts_violation_threshold, candidate_cuts)

        # Create mapping of user cut violations for quick lookup
        user_cut_violations = Dict(cut.k_idx => cut.violation / cut.norm for cut in cuts)
        
        # Filter mixing cuts - only keep those with higher normalized violation than original user cut
        candidate_cuts = filter(x -> x.normalized_violation > user_cut_violations[x.k_idx] * (1 + config.user_mixing_cuts_normed_violation_rel_threshold), candidate_cuts)
        
        sort!(candidate_cuts, by=x -> x.normalized_violation, rev=true)

        for _ in 1:max_mixing_cuts_submit_per_round
            if isempty(candidate_cuts)
                break
            end

            best_cut = pop!(candidate_cuts)

            # Add mixing cut with proper RHS
            vars = vcat(solver.x, solver.z)
            mixing_con = @build_constraint(
                sum(best_cut.coefficients[i] * vars[i] for i in 1:length(vars)) >= best_cut.constant
            )

            MOI.submit(solver.model, MOI.UserCut(cb_data), mixing_con)

            # Track the scenario index
            push!(submitted_scenario_indices, best_cut.k_idx)

            if config.mixing_print_level >= 1
                print_mixing_inequality(solver.num_x, best_cut.coefficients, best_cut.constant, best_cut.k_idx, "user")
                println("Violation: ", round(best_cut.violation, digits=6))
                println("Normalized violation: ", round(best_cut.normalized_violation, digits=6))
            end

            if config.mixing_print_level >= 3
                println("Adding user mixing cut:", mixing_con)
            end

            # Update statistics
            user_mixing_cuts_added += 1
            solver.stats.user_mixing_cuts_added += 1


            if is_root
                solver.stats.root_stats["user_mixing_cuts_added"] += 1
            else
                solver.stats.tree_stats["user_mixing_cuts_added"] += 1
            end
        end
    end

    println("submitted ", user_mixing_cuts_added, " user mixing cuts. Total: ", solver.stats.root_stats["user_mixing_cuts_added"] + solver.stats.tree_stats["user_mixing_cuts_added"])

       # Update timing statistics
    solver.stats.user_mixing_cuts_time += time() - user_mixing_start_time
   
    return user_mixing_cuts_added, submitted_scenario_indices
end