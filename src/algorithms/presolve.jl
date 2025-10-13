# src/algorithms/presolve.jl


"""
Find all scenarios dominated by scenario k
D(k) := {s ∈ I : E_s ⊆ E_k}
"""
function find_dominated_scenarios(instance::PPMPInstance, k::Int)
    dominated = Set{Int}()
    failed_edges_k = Set(instance.failed_edges[k])
    
    for s in 1:length(instance.scenarios)
        # A scenario dominates itself
        if s == k
            push!(dominated, s)
            continue
        end
        
        # Check if failed edges in k are a subset of failed edges in s
        # If true, then k dominates s
        failed_edges_s = Set(instance.failed_edges[s])
        if issubset(failed_edges_k, failed_edges_s)
            push!(dominated, s)
        end
    end
    
    return dominated
end

"""
Create new PPMPInstance with removed scenarios
"""
function create_reduced_instance(instance::PPMPInstance,
                               fixed_scenarios::Set{Int},
                               removed_scenarios::Set{Int},
                               new_epsilon::Float64)
    # Keep all scenarios except the removed ones
    kept_scenario_indices = setdiff(1:length(instance.scenarios), removed_scenarios)

    # Create new instance maintaining both scenarios and failed_edges
    return PPMPInstance(instance, kept_scenario_indices, new_epsilon)
end


"""
Print detailed presolve analysis and results
"""
function print_presolve_results(instance::PPMPInstance,
                              dominance_map::Dict{Int,Set{Int}},
                              original_fixed::Set{Int},
                              removed_scenarios::Set{Int},
                              fixed_scenarios::Vector{Int},
                              kept_scenarios::Vector{Int},
                              scenario_map::Dict{Int,Int},
                              new_epsilon::Float64)
    println("\nPresolve Analysis Results")
    println("=======================")
    
    # Basic statistics
    println("\n1. Basic Information:")
    println("-------------------")
    println("Total scenarios: ", length(instance.scenarios))
    println("Scenarios removed: ", length(removed_scenarios))
    println("Scenarios fixed: ", length(fixed_scenarios))
    println("Scenarios retained: ", length(instance.scenarios) - length(removed_scenarios))
    println("Original ε: ", instance.epsilon)
    if !isempty(removed_scenarios)
        println("New ε: ", new_epsilon, " (increased by ", 
                sum(instance.probabilities[k] for k in removed_scenarios), ")")
    end
    
    # Dominance relationships
    println("\n2. Dominance Analysis:")
    println("--------------------")
    for k in 1:length(instance.scenarios)
        dominated_prob = sum(instance.probabilities[i] for i in dominance_map[k])
        println("\nScenario $k:")
        println("  Probability: ", round(instance.probabilities[k], digits=4))
        println("  Dominates scenarios: ", sort(collect(dominance_map[k])))
        println("  Total dominated probability: ", round(dominated_prob, digits=4))
        println("  P(D($k)\\{$k}): ", 
                round(dominated_prob - instance.probabilities[k], digits=4))
        println("  Status: ", if k in removed_scenarios
            "REMOVED (ε-subfixed)"
        elseif k in original_fixed
            "FIXED (ε-fixed)"
        else
            "retained"
        end)
    end
    
    # Scenario mapping details
    println("\n3. Scenario Mapping:")
    println("------------------")
    println("Original → New indices:")
    for (old_idx, new_idx) in sort(collect(scenario_map))
        status = if old_idx in original_fixed
            "(FIXED)"
        else
            ""
        end
        println("  $old_idx → $new_idx $status")
    end
    
    # Removed scenarios details
    if !isempty(removed_scenarios)
        println("\n4. Removed Scenarios (ε-subfixed):")
        println("--------------------------------")
        for k in sort(collect(removed_scenarios))
            dominated_without_k = sum(instance.probabilities[i] for i in setdiff(dominance_map[k], Set([k])))
            println("\nScenario $k:")
            println("  Probability: ", round(instance.probabilities[k], digits=4))
            println("  P(D($k)\\{$k}): ", round(dominated_without_k, digits=4))
            println("  Dominance condition: ", round(dominated_without_k, digits=4), 
                    " > ", instance.epsilon, " (ε)")
        end
    end
    
    # Fixed scenarios details
    if !isempty(fixed_scenarios)
        println("\n5. Fixed Scenarios (ε-fixed):")
        println("----------------------------")
        println("Original indices: ", sort(collect(original_fixed)))
        println("New indices: ", fixed_scenarios)
        for k in sort(collect(original_fixed))
            dominated_prob = sum(instance.probabilities[i] for i in dominance_map[k])
            println("\nScenario $k → $(scenario_map[k]):")
            println("  Probability: ", round(instance.probabilities[k], digits=4))
            println("  P(D($k)): ", round(dominated_prob, digits=4))
            println("  Dominance condition: ", round(dominated_prob, digits=4), 
                    " > ", instance.epsilon, " (ε)")
        end
    end
    
    # Final model size
    println("\n6. Final Model Size:")
    println("-----------------")
    println("Original scenarios: ", length(instance.scenarios))
    println("Retained scenarios: ", length(kept_scenarios))
    println("Fixed variables: ", length(fixed_scenarios))
    println("Free variables: ", length(kept_scenarios) - length(fixed_scenarios))
end

"""
Apply presolve reductions to PPMP instance
Returns (reduced_instance, presolve_info)
"""
# function ppmp_presolve(instance::PPMPInstance)

#     presolve_time_start = time()
#     # Step 1: Build dominance relationships (unchanged)
#     dominance_map = Dict{Int,Set{Int}}()
#     for k in 1:length(instance.scenarios)
#         dominance_map[k] = find_dominated_scenarios(instance, k)
#     end
    
#     # Step 2: Find ε-fixed and ε-subfixed scenarios (unchanged)
#     original_fixed = Set{Int}()
#     removed_scenarios = Set{Int}()
    
#     for k in 1:length(instance.scenarios)
#         dominated_prob = sum(instance.probabilities[i] for i in dominance_map[k])
        
#         if dominated_prob - instance.probabilities[k] > instance.epsilon + 1e-6
#             # k is ε-subfixed
#             push!(removed_scenarios, k)
#         elseif dominated_prob > instance.epsilon + 1e-6
#             # k is ε-fixed but not ε-subfixed
#             push!(original_fixed, k)
#         end
#     end
    
#     # Step 3: Get kept scenarios and create mapping
#     kept_scenarios = sort(collect(setdiff(1:length(instance.scenarios), removed_scenarios)))
#     scenario_map = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(kept_scenarios))
    
#     # Convert fixed scenarios to new indices
#     fixed_scenarios = Int[scenario_map[k] for k in original_fixed]
#     sort!(fixed_scenarios)  # Ensure sorted order
    
#     println("removed: ", length(removed_scenarios))
#     println("fixed: ", length(original_fixed))

#     # Calculate new epsilon
#     new_epsilon = isempty(removed_scenarios) ?  instance.epsilon : instance.epsilon + sum(instance.probabilities[k] for k in removed_scenarios)
    
#     # Create presolve statistics
#     presolve_time = time() - presolve_time_start
#     presolve_stats = PresolveStats(presolve_time, instance, dominance_map, original_fixed, removed_scenarios, new_epsilon)
    
#     # print_presolve_results(instance, dominance_map, original_fixed, removed_scenarios, fixed_scenarios, kept_scenarios, scenario_map, new_epsilon)

#     # Create reduced instance (which will use sequential indices)
#     reduced_instance = PPMPInstance(instance, kept_scenarios, new_epsilon)


    
#      # Create presolve info with stats
#     presolve_info = PresolveInfo(fixed_scenarios, removed_scenarios, dominance_map, new_epsilon, scenario_map, presolve_stats)


#     return reduced_instance, presolve_info
# end

function ppmp_presolve(instance::PPMPInstance)

    presolve_time_start = time()
    
    # Build dominance relationships
    dominance_map = Dict{Int,Set{Int}}()
    for k in 1:length(instance.scenarios)
        dominance_map[k] = find_dominated_scenarios(instance, k)
    end
    
    # Find ε-fixed and ε-subfixed scenarios
    fixed_scenarios = Int[]
    subfixed_scenarios = Int[]
    
    for k in 1:length(instance.scenarios)
        dominated_prob = sum(instance.probabilities[i] for i in dominance_map[k])
        
        if dominated_prob - instance.probabilities[k] > instance.epsilon + 1e-6
            # k is ε-subfixed
            push!(subfixed_scenarios, k)
        elseif dominated_prob > instance.epsilon + 1e-6
            # k is ε-fixed but not ε-subfixed
            push!(fixed_scenarios, k)
        end
    end
    
    # Calculate new epsilon
    new_epsilon = if isempty(subfixed_scenarios)
        instance.epsilon  # If no subfixed scenarios, keep original epsilon
    else 
        instance.epsilon + sum(instance.probabilities[k] for k in subfixed_scenarios)
    end
        
    # Create presolve statistics
    presolve_time = time() - presolve_time_start
    presolve_stats = PresolveStats(presolve_time, instance, dominance_map, 
                                 Set(fixed_scenarios), Set(subfixed_scenarios), new_epsilon)
    
    # Return original instance with presolve info
    return instance, PresolveInfo(fixed_scenarios, subfixed_scenarios, 
                                dominance_map, new_epsilon, presolve_stats)
end