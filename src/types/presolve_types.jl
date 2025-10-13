# src/types/presolve_types.jl

# src/types/presolve_types.jl

"""
Structure to hold statistics from scenario merging analysis
"""
struct MergeStats
    # Basic counts
    total_scenarios::Int             
    num_unique_scenarios::Int         
    num_duplicated::Int              
    
    # Group statistics
    max_group_size::Int
    min_group_size::Int
    avg_group_size::Float64
    median_group_size::Float64
    
    # Group details
    scenario_groups::Vector{Tuple{Vector{Int}, Int, Float64}}  # [(group_indices, size, probability), ...]
    
    # Time statistics
    merge_time::Float64

    function MergeStats(
        num_original_scenarios::Int,
        merged_scenarios::Vector{Vector{Tuple{Int,Int}}},
        merged_probabilities::Vector{Float64},
        scenario_groups::Dict{Vector{Tuple{Int,Int}}, Vector{Int}},
        merge_time::Float64
    )
        # Calculate basic counts
        num_unique = length(merged_scenarios)
        num_duplicated = num_original_scenarios - num_unique
        
        # Calculate group size statistics
        group_sizes = [length(group) for group in values(scenario_groups)]
        
        # Handle empty scenario groups case
        if isempty(group_sizes)
            max_size = 1  # or 0, depending on your preference
            min_size = 1  # or 0
            avg_size = 1.0  # or 0.0
            median_size = 1.0  # or 0.0
        else
            max_size = maximum(group_sizes)
            min_size = minimum(group_sizes)
            avg_size = mean(group_sizes)
            median_size = median(group_sizes)
        end
        
        # Create group info tuples
        scenario_to_prob = Dict(zip(merged_scenarios, merged_probabilities))
        groups_info = Vector{Tuple{Vector{Int}, Int, Float64}}()
        for (scenario, indices) in scenario_groups
            push!(groups_info, (
                indices,                    # group members
                length(indices),            # group size
                scenario_to_prob[scenario]  # group probability
            ))
        end
        # Sort by group size
        sort!(groups_info, by=x -> x[2], rev=true)
        
        new(
            num_original_scenarios,
            num_unique,
            num_duplicated,
            max_size,
            min_size,
            avg_size,
            median_size,
            groups_info,
            merge_time
        )
    end
end

"""
Structure to hold statistics from presolve analysis
"""
struct PresolveStats
    presolve_time::Float64         # Time spent in presolve

    total_scenarios::Int             
    num_removed::Int                 
    num_fixed::Int                   
    num_retained::Int               
    
    original_epsilon::Float64        
    new_epsilon::Float64            
    epsilon_increase::Float64       
    
    removed_probability::Float64    
    fixed_probability::Float64      
    retained_probability::Float64   
    
    fixed_scenario_probs::Vector{Tuple{Int,Float64}}    
    removed_scenario_probs::Vector{Tuple{Int,Float64}}  
    
    # Dominance statistics
    max_dominance_prob::Float64     # Maximum probability dominated
    min_dominance_prob::Float64     # Minimum probability dominated
    avg_dominance_prob::Float64     # Average probability dominated
    
    # New dominance count statistics
    max_dominating_scenario::Int    # The scenario that dominates the most others
    avg_dominated_count::Float64    # Average number of scenarios dominated per scenario
end

"""
Create PresolveStats from presolve results and instance data
"""
function PresolveStats(presolve_time::Float64,
                    instance::PPMPInstance,
                    dominance_map::Dict{Int,Set{Int}},
                    original_fixed::Set{Int},
                    removed_scenarios::Set{Int},
                    new_epsilon::Float64)

    # Basic counts
    total_scenarios = length(instance.scenarios)
    num_removed = length(removed_scenarios)
    num_fixed = length(original_fixed)
    num_retained = total_scenarios - num_removed
    
    # Epsilon changes
    epsilon_increase = isempty(removed_scenarios) ? 0.0 :
                      sum(instance.probabilities[k] for k in removed_scenarios)
    
    # Probability sums
    removed_probability = isempty(removed_scenarios) ? 0.0 :
                         sum(instance.probabilities[k] for k in removed_scenarios)
    fixed_probability = isempty(original_fixed) ? 0.0 :
                       sum(instance.probabilities[k] for k in original_fixed)
    retained_probability = 1.0 - removed_probability
    
    # Store detailed probability information
    fixed_scenario_probs = isempty(original_fixed) ? Tuple{Int,Float64}[] :
                          [(k, instance.probabilities[k]) for k in sort(collect(original_fixed))]
    removed_scenario_probs = isempty(removed_scenarios) ? Tuple{Int,Float64}[] :
                            [(k, instance.probabilities[k]) for k in sort(collect(removed_scenarios))]
    
    # Compute dominance statistics
    dominance_probs = Float64[]
    for k in 1:total_scenarios
        if !(k in removed_scenarios)  # Only consider retained scenarios
            dominated_prob = sum(instance.probabilities[i] for i in dominance_map[k])
            push!(dominance_probs, dominated_prob)
        end
    end
    
    max_dominance_prob = isempty(dominance_probs) ? 0.0 : maximum(dominance_probs)
    min_dominance_prob = isempty(dominance_probs) ? 0.0 : minimum(dominance_probs)
    avg_dominance_prob = isempty(dominance_probs) ? 0.0 : mean(dominance_probs)
    
    # Compute maximum number of dominated scenarios
    max_dominated_count = isempty(dominance_map) ? 0 : 
                        maximum(length(dom_set) for dom_set in values(dominance_map))  # -1 to exclude self

    # Average number of dominated scenarios
    avg_dominated_count = isempty(dominance_map) ? 0.0 : 
                        mean(length(dom_set) for dom_set in values(dominance_map))
    PresolveStats(
        presolve_time, 
        total_scenarios,
        num_removed,
        num_fixed,
        num_retained,
        instance.epsilon,
        new_epsilon,
        epsilon_increase,
        removed_probability,
        fixed_probability,
        retained_probability,
        fixed_scenario_probs,
        removed_scenario_probs,
        max_dominance_prob,
        min_dominance_prob,
        avg_dominance_prob,
        max_dominated_count,
        avg_dominated_count   
    )
end

"""
Structure to hold presolve information
"""
# struct PresolveInfo
#     fixed_scenarios::Vector{Int}     # Indices in the REDUCED instance
#     removed_scenarios::Set{Int}      # Original indices that were removed
#     dominance_map::Dict{Int,Set{Int}}
#     new_epsilon::Float64
#     scenario_map::Dict{Int,Int}      # Maps original indices to new indices
#     stats::PresolveStats            # Statistical information about presolve
# end
struct PresolveInfo
    fixed_scenarios::Vector{Int}     # Indices of epsilon-fixed scenarios 
    subfixed_scenarios::Vector{Int}  # Indices of epsilon-subfixed scenarios (previously removed)
    dominance_map::Dict{Int,Set{Int}}
    new_epsilon::Float64
    stats::PresolveStats
end