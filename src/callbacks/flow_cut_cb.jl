# src/callbacks/flow_cut_cb.jl

using JuMP
using CPLEX

"""
Store a cut to solver's cutpool when it's accepted
"""
function store_cut(cutpool::CutPool, coefficients::Vector{Float64},
                    violation::Float64, norm::Float64, fingerprint::String, is_lazy::Bool, in_round::Int, depth::Int)

    # Create new cut
    cut = StoredCut(coefficients, 0.0, fingerprint, violation, norm, is_lazy, in_round, depth)
    
    # Update pool
    push!(cutpool.cuts, cut)
    push!(cutpool.fingerprints, fingerprint)
    cutpool.pool_size += 1
    cutpool.fingerprint_to_index[fingerprint] = cutpool.pool_size
end

"""
Update an existing cut in the pool using pre-found index
"""
function update_cut(cutpool::CutPool, violation::Float64, fingerprint::String, in_round::Int, depth::Int)

    # Get index directly from dictionary
    cut_idx = cutpool.fingerprint_to_index[fingerprint]
    cut = cutpool.cuts[cut_idx]
    
    # Update the existing cut's fields
    cut.violation = violation
    cut.in_round = in_round
    cut.depth = depth

    # Update fingerprint hit counter
    cutpool.fingerprint_hit += 1
end

"""
Flow cuts callback with updated statistics tracking
"""
function flow_user_cuts_callback(cb_data, solver::FlowCutSolver, config::PPMPConfig, glb_pool::FlowCutGlbPool)

    callback_type = "user"

    callback_start_time = time()

    try

        user_cuts_added = solver.stats.root_stats["user_cuts_added"] + solver.stats.tree_stats["user_cuts_added"]
        if user_cuts_added >= config.max_total_user_cuts
            return
        end
        
        
        # Get node information (node count, depth)
        node_count = Ref{Cint}()
        node_depth = Ref{Cint}()
        CPXcallbackgetinfoint(cb_data, CPXCALLBACKINFO_NODECOUNT, node_count) # Node count
        if config.callback_print_level >= 3
            println("node: ", node_count[])
        end
        CPXcallbackgetinfoint(cb_data, CPXCALLBACKINFO_NODEDEPTH, node_depth) # Node depth
        is_root = node_count[] == 0

        depth = Int(node_depth[])

        # if !is_root && node_depth[] % config.tree_user_frequency != 0
        #     return
        # end

        # Track rounds for current node
        if node_count[] != solver.stats.cur_node_idx
            solver.stats.cur_node_idx = node_count[]
            solver.stats.user_cuts_cur_node = 0
            solver.stats.user_rounds_cur_node = 0
        end

        # Get depth scaling factor (only for tree nodes)
        depth_factor = is_root ? 1.0 : get_depth_scaling_factor(Int(node_depth[]), config.scaling_type, config.depth_decay_rate, config.min_scaling_factor)

            # Check if we've exceeded rounds per node
        max_rounds = is_root ? config.root_user_max_rounds_per_node : max(1, ceil(Int, config.tree_user_max_rounds_per_node * depth_factor))
        if solver.stats.user_rounds_cur_node >= max_rounds
            return
        end
        
        # Get parameters using ternary operators for clarity
        max_cuts_per_node = is_root ? config.root_user_max_cuts_per_node : round(Int, config.tree_user_max_cuts_per_node * depth_factor) # if too deep, we don't add cuts
        
        already_rounds = solver.stats.user_rounds_cur_node + 1  # +1 to avoid division by zero
        max_cuts_per_round = is_root ? config.root_user_max_cuts_per_round : round(Int, config.tree_user_max_cuts_per_round * (depth_factor / already_rounds))
        
        max_scenarios = is_root ? config.root_user_max_scenarios : config.tree_user_max_scenarios
        min_violation = is_root ? config.root_user_min_violation : config.tree_user_min_violation
        
        remaining_node_cuts = max_cuts_per_node - solver.stats.user_cuts_cur_node
        max_cuts_this_round = min(max_cuts_per_round, remaining_node_cuts)

        is_whole_search = is_root ? config.root_user_scenario_whole_search : config.tree_user_scenario_whole_search

        if max_cuts_per_round <=0 || max_cuts_per_node <= 0
            
            if config.callback_print_level >= 1
                location = is_root ? "root node    " : @sprintf("node %-4d | depth %-2d", node_count[], node_depth[])
                println(@sprintf("Added lazy cuts at %-20s | cuts %3d/%-3d | round %2d/%-2d | min vio: %.6f",
                                location,
                                0,
                                max_cuts_this_round,
                                solver.stats.user_rounds_cur_node,
                                max_rounds,
                                0.0))
            end
            return 
        end

        
            # Check if we've exceeded cuts for this node
        if solver.stats.user_cuts_cur_node >= max_cuts_per_node
            return
        end

                
        # Get current solution
        # CPLEX.load_callback_variable_primal(cb_data, context_id)
        x_val = [callback_value(cb_data, x) for x in solver.x]
        z_val = [callback_value(cb_data, z) for z in solver.z]
        # vars = vcat(x_val, z_val)
        vars = vcat(solver.x, solver.z)

        # record the relaxation solution at root node
        if is_root
            
            # println("Update root fractional solution")
            glb_pool.root_fraction_sol = vcat(x_val, z_val)
            glb_pool.root_fraction_obj = sum(vcat(solver.instance.costs, zeros(length(z_val))) .* glb_pool.root_fraction_sol)
            # println("root_fraction_obj: ", glb_pool.root_fraction_obj)
            # Update best relaxation solution
            if glb_pool.root_fraction_obj > glb_pool.best_fraction_obj + 1e-6 || isempty(glb_pool.best_fraction_sol)
                # println("Update best fractional solution")
                glb_pool.best_fraction_obj = glb_pool.root_fraction_obj
                glb_pool.best_fraction_sol = glb_pool.root_fraction_sol
            end
        end


        # Find violated cuts
        cuts = find_violated_cuts(solver, x_val, z_val,
                                max_cuts=min(max_cuts_this_round, solver.num_z),
                                max_scenarios=max_scenarios,  # for user cuts, we may use lesser cuts
                                min_violation=min_violation,
                                is_whole_search=is_whole_search)
        if config.callback_print_level >= 1 && isempty(cuts)
            println("No user cuts found at ",
                    is_root ? "root node" : "node $(node_count[]) (depth $(node_depth[]))")
        end

        """Mixing cuts"""
        # First submit mixing cuts and get covered scenarios
        mixing_cuts_added, covered_scenarios = process_mixing_cuts_user!(cb_data, solver, config, 
                                                                       cuts, x_val, z_val, is_root)

        """User Cuts"""
        cuts_added = 0
        if config.user_cuts_enabled && !isempty(cuts) 
            for cut in cuts

                # Skip if scenario already covered by mixing cuts
                if cut.k_idx in covered_scenarios
                    continue
                end

                fingerprint = join(round.(cut.coefficients), ",")
                # if fingerprint ∉ solver.cutpool.fingerprints  # we turn if off we use CPX_USECUT_FILTER
        
                con = @build_constraint(sum(cut.coefficients[i] * vars[i] for i in 1:length(vars)) >= 0)
                MOI.submit(solver.model, MOI.UserCut(cb_data), con)

                # we turn it off for since we use CPX_USECUT_FILTER
                # # Store the user cuts to solver's cutpool
                # store_cut(solver.cutpool, cut.coefficients, is_root, cut.violation, cut.norm, fingerprint, false, solver.solve_rounds) # truee the user cut to solver's cutpool
                
                # Store the user cut to global cutpool
                if solver.restart_flag # if we are not in restart mode (final solve), we don't store the cut
                    if fingerprint ∉ glb_pool.user_pool.fingerprints
                        store_cut(glb_pool.user_pool, cut.coefficients, cut.violation, cut.norm, fingerprint, false, solver.solve_rounds, depth)
                    else 
                        update_cut(glb_pool.user_pool, cut.violation, fingerprint, solver.solve_rounds, depth)
                    end
                end

                # Update statistics
                cuts_added += 1
                solver.stats.total_cuts_added += 1
                solver.stats.user_cuts_cur_node += 1
                # push!(solver.stats.cut_violations, cut.violation)
                
                # Track scenario contribution
                for k in 1:solver.num_z
                    if abs(cut.coefficients[solver.num_x + k]) > 1e-6
                        solver.stats.cuts_per_scenario[k] = 
                            get(solver.stats.cuts_per_scenario, k, 0) + 1
                        break
                    end
                end
                
                # Update location-specific statistics
                if is_root
                    solver.stats.root_stats["user_cuts_added"] += 1
                else
                    solver.stats.tree_stats["user_cuts_added"] += 1
                    solver.stats.tree_stats["nodes_processed"] += 1
                end
                # else
                #     if config.callback_print_level >= 1
                #         println(callback_type, " fingerprint already exists, violation: ", cut.violation)
                #         solver.stats.fingerprint_already_exists += 1
                #     end
                # end
            end
            # Update round counter if any cuts were found
            if !isempty(cuts)
                solver.stats.user_rounds_cur_node += 1
            end
        end

        # """Mixing cuts"""
        # user_mixing_cuts_added = 0
        # if config.user_mixing_cuts_enabled && !isempty(cuts) 
        #     user_mixing_cuts_added = process_mixing_cuts_user!(cb_data, solver, config, cuts, x_val, z_val, is_root)
        # end

        # Update callback statistics
        solver.stats.user_callbacks += 1
        solver.stats.user_callbacks_time += time() - callback_start_time


        if config.callback_print_level >= 1 && cuts_added > 0
            location = is_root ? "root node    " : @sprintf("node %-4d | depth %-2d", node_count[], node_depth[])
            println(@sprintf("Added user cuts at %-20s | cuts %3d/%-3d | round %2d/%-2d | min vio: %.6f",
                            location,
                            cuts_added,
                            max_cuts_this_round,
                            solver.stats.user_rounds_cur_node,
                            max_rounds,
                            minimum([cut.violation for cut in cuts])))
        end
        if config.callback_print_level >= 1
            println("depth factor: ", depth_factor)
        end                

    catch e
        println("Error in user cut callback: ", e)
        println(stacktrace())
    end
    
    
    # Update timing statistics
    solver.stats.total_callbacks += 1
    solver.stats.total_callbacks_time += time() - callback_start_time
    
end

"""
Flow cuts callback with updated statistics tracking
"""
function flow_lazy_cons_callback(cb_data, solver::FlowCutSolver, config::PPMPConfig, glb_pool::FlowCutGlbPool)
    # if solver.stats.total_cuts_added >= config.max_total_cuts
    #     return
    # end
    callback_type = "lazy"

    callback_start_time = time()
        
    try
        # Get node information (node count, depth)
        node_count = Ref{Cint}()
        node_depth = Ref{Cint}()
        CPXcallbackgetinfoint(cb_data, CPXCALLBACKINFO_NODECOUNT, node_count)
        CPXcallbackgetinfoint(cb_data, CPXCALLBACKINFO_NODEDEPTH, node_depth)
        is_root = node_count[] == 0

        depth = Int(node_depth[])
        

        max_lazy_cons = is_root ? config.root_lazy_cuts_per_node : config.tree_lazy_cuts_per_node
        is_whole_search = is_root ? config.root_lazy_scenario_whole_search : config.tree_lazy_scenario_whole_search
        
        x_val = [callback_value(cb_data, x) for x in solver.x]
        z_val = [callback_value(cb_data, z) for z in solver.z]
        # vars = vcat(x_val, z_val)
        vars = vcat(solver.x, solver.z)

        # Find violated cuts    # record the relaxation solution
        cuts = find_violated_cuts(solver, x_val, z_val,
                                max_cuts=min(max_lazy_cons, solver.num_z),
                                max_scenarios=config.lazy_max_scenarios,
                                min_violation=config.lazy_min_violation,
                                is_whole_search=is_whole_search)
        
        if config.callback_print_level >= 1
            if isempty(cuts)
                println("No lazy cuts found at ",
                        is_root ? "root node" : "node $(node_count[]) (depth $(node_depth[]))")
            else
                println("Found $(length(cuts)) lazy cuts at ",
                        is_root ? "root node" : "node $(node_count[]) (depth $(node_depth[]))")
            end
            
        end
        
        cuts_added = 0
        for cut in cuts
            # fingerprint = join(string(cut.coefficients), ",") # we turn it off since we use CPX_USECUT_FILTER
            # if fingerprint ∉ solver.cutpool.fingerprints
            con = @build_constraint(sum(cut.coefficients[i] * vars[i] for i in 1:length(vars)) >= 0)
            
            MOI.submit(solver.model, MOI.LazyConstraint(cb_data), con)

            # we turn it off since we use CPX_USECUT_FILTER
            # Store the lazy constraint for restart
            # store_cut(solver.cutpool, cut.coefficients, is_root, cut.violation, cut.norm, fingerprint, true, solver.solve_rounds) # Store the user cut to solver's cutpool
            
            # Store the user cut to global cutpool
            fingerprint = join(round.(cut.coefficients), ",")
            if solver.restart_flag
                if fingerprint ∉ glb_pool.lazy_pool.fingerprints
                    store_cut(glb_pool.lazy_pool, cut.coefficients, cut.violation, cut.norm, fingerprint, true, solver.solve_rounds, depth)
                else 
                    update_cut(glb_pool.lazy_pool, cut.violation, fingerprint, solver.solve_rounds, depth)
                end
            end

            # Update statistics
            cuts_added += 1
            solver.stats.total_cuts_added += 1
            
            # Update location-specific statistics
            if is_root
                solver.stats.root_stats["lazy_cuts_added"] += 1
                solver.stats.root_stats["max_violation"] = 
                    max(solver.stats.root_stats["max_violation"], cut.violation)
            else
                solver.stats.tree_stats["lazy_cuts_added"] += 1
            end
            # else
            #     if config.callback_print_level >= 1
            #         println("$(callback_type) fingerprint already exists, violation: ", cut.violation)
            #     end
            # end
        end

        # Update callback statistics
        solver.stats.lazy_callbacks += 1
        solver.stats.lazy_callbacks_time += time() - callback_start_time

        if config.callback_print_level >= 1 && cuts_added > 0
            println("Added $cuts_added lazy cuts at ",
                    is_root ? "root node" : "node $(node_count[]) (depth $(node_depth[]))",
                    " min vio: ", round(minimum([cut.violation for cut in cuts]), digits=6))
        end

        """Mixing"""
        lazy_mixing_cuts_added = 0
        if !isempty(cuts) && config.lazy_mixing_cuts_enabled
            lazy_mixing_cuts_added = process_mixing_cuts_lazy!(cb_data, solver, config, cuts, x_val, z_val, is_root)
        end
        if config.callback_print_level >= 1 && lazy_mixing_cuts_added > 0
            println("Added $lazy_mixing_cuts_added mixing cuts at ",
                    is_root ? "root node" : "node $(node_count[]) (depth $(node_depth[]))")
        end
        
    catch e
        println("Error in lazy constraint callback: ", e)
        println(stacktrace())
    end

    # Update timing statistics
    solver.stats.total_callbacks += 1
    solver.stats.total_callbacks_time += time() - callback_start_time
end