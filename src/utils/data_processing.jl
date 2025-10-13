
# src/utils/data_processing.jl

# Data processing utilities

"""
Load instance from JSON file with scenario merging and group statistics
Returns:
- RawInstanceData: The instance data with merged scenarios
- Vector{Float64}: The merged scenario probabilities
"""
function load_instance_from_json(filename::String)
    data = JSON.parsefile(filename)
    
    # Extract base data
    n = data["instance_info"]["n"]
    edges = [(e[1], e[2]) for e in data["graph"]["edges"]]
    num_scenarios = data["instance_info"]["num_scenarios"]
    
    
    # Handle empty arrays with explicit typing
    failed_edges = Vector{Vector{Tuple{Int,Int}}}()
    for scenario in data["scenarios"]["failed_edges"]
        if isempty(scenario)
            # For empty scenarios, push empty vector with correct type
            push!(failed_edges, Tuple{Int,Int}[])
        else
            # For non-empty scenarios, convert to correctly typed tuples
            push!(failed_edges, [(e[1], e[2]) for e in scenario])
        end
    end
    
    # Create equal probabilities initially
    probabilities = fill(1.0/num_scenarios, num_scenarios)
    
    merge_start_time = time()
    # Merge identical scenarios and get group information
    merged_scenarios, merged_probabilities, scenario_groups = merge_identical_scenarios(failed_edges, probabilities)    # Create merge stats

    merge_time = time() - merge_start_time
    merge_stats = MergeStats(num_scenarios, merged_scenarios, merged_probabilities, 
                           scenario_groups, merge_time)

    # Print statistics
    print_scenario_merge_statistics(merge_stats)

    return RawInstanceData(n, edges, length(merged_scenarios), merged_scenarios), 
           merged_probabilities,
           merge_stats
end


"""
Merge identical scenarios and track grouping information
Returns:
- merged_scenarios: Vector of unique scenarios
- merged_probabilities: Vector of summed probabilities for each unique scenario
- scenario_groups: Dictionary mapping fingerprint to list of original scenario indices
"""
function merge_identical_scenarios(scenarios::Vector{Vector{Tuple{Int,Int}}}, 
                                 probabilities::Vector{Float64})
    # Create scenario fingerprints for comparison
    scenario_fingerprints = [sort(collect(scenario)) for scenario in scenarios]
    
    # Track unique scenarios, their probabilities, and grouping information
    unique_scenarios = Dict{Vector{Tuple{Int,Int}}, Float64}()
    scenario_groups = Dict{Vector{Tuple{Int,Int}}, Vector{Int}}()
    
    # Merge scenarios and track groups
    for (idx, scenario) in enumerate(scenario_fingerprints)
        fingerprint = scenario
        if haskey(unique_scenarios, fingerprint)
            unique_scenarios[fingerprint] += probabilities[idx]
            push!(scenario_groups[fingerprint], idx)
        else
            unique_scenarios[fingerprint] = probabilities[idx]
            scenario_groups[fingerprint] = [idx]
        end
    end
    
    # Convert scenarios and probabilities to vectors
    merged_scenarios = collect(keys(unique_scenarios))
    merged_probabilities = collect(values(unique_scenarios))
    
    # Normalize probabilities
    prob_sum = sum(merged_probabilities)
    if !isapprox(prob_sum, 1.0, atol=1e-6)
        println("Normalizing probabilities")
        merged_probabilities ./= prob_sum
    end

    # Filter out groups of size 1
    filter_scenario_groups = Dict{Vector{Tuple{Int,Int}}, Vector{Int}}()
    for (scenario, group) in scenario_groups
        if length(group) > 1
            filter_scenario_groups[scenario] = group
        end
    end
    
    return merged_scenarios, merged_probabilities, filter_scenario_groups
end


"""
Print detailed statistics about scenario merging
"""
function print_scenario_merge_statistics(merge_stats::MergeStats)
    # Print original instance info
    println("\nOriginal instance:")
    println("Number of scenarios: ", merge_stats.total_scenarios)
    println("Probability per scenario: ", round(1.0/merge_stats.total_scenarios, digits=4))
    
    # Print detailed merging results
    println("\nScenario Group Analysis:")
    println("------------------------")
    println("Number of unique scenario groups: ", merge_stats.num_unique_scenarios)
    
    # Report group sizes and members
    println("\nGroup size distribution:")
    for (i, (group, size, prob)) in enumerate(merge_stats.scenario_groups)
        println("Group $i: $size scenarios (original indices: $group)")
        println("  Combined probability: $(round(prob, digits=4))")
    end
    
    # Print summary statistics
    println("\nSummary:")
    println("- Original scenarios: ", merge_stats.total_scenarios)
    println("- Unique scenarios after merging: ", merge_stats.num_unique_scenarios)
    println("- Number of duplicated scenarios: ", merge_stats.num_duplicated)
    
    println("\nGroup Size Statistics:")
    println("- Maximum group size: ", merge_stats.max_group_size, " scenarios")
    println("- Minimum group size: ", merge_stats.min_group_size, " scenarios")
    println("- Average group size: ", round(merge_stats.avg_group_size, digits=2), " scenarios")
    println("- Median group size: ", merge_stats.median_group_size, " scenarios")
    println("\nMerge processing time: ", round(merge_stats.merge_time, digits=3), " seconds")
end