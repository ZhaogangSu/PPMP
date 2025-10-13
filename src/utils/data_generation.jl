# src/utils/data_generation.jl

using JSON
using Graphs
using Random
using BipartiteMatching


"""
Generate random bipartite graph with valid scenarios
Returns:
- edges: Vector of edges in the final graph
- failed_edges: Vector of vectors containing scenario failures
"""
function generate_bipartite_with_scenarios(n::Int, 
                                         density::Float64,
                                         num_scenarios::Int,
                                         failure_prob::Float64;
                                         max_graph_attempts::Int=1000000,
                                         max_scenario_attempts::Int=100)
    @assert 0 < density <= 1 "Density must be between 0 and 1"
    @assert 0 <= failure_prob <= 1 "Failure probability must be between 0 and 1"
    
    for graph_attempt in 1:max_graph_attempts
        # Generate initial edges
        edges = [(i,j) for i in 1:n for j in 1:n if rand() <= density]
        
        # Create adjacency matrix using BitArray for efficiency
        adj = falses(n, n)
        [adj[i,j] = true for (i,j) in edges]
        
        # Check if original graph has a perfect matching
        matching, matched = findmaxcardinalitybipartitematching(adj)
        if length(matching) != n
            continue  # Try another graph
        end
        
        # Try to generate valid scenarios
        for scenario_attempt in 1:max_scenario_attempts
            failed_edges = Vector{Vector{Tuple{Int,Int}}}(undef, num_scenarios)
            valid_scenarios = true
            
            # Generate and test each scenario
            for k in 1:num_scenarios
                # # Generate failures
                # # failed_edges[k] = [(i,j) for (i,j) in edges if rand() <= failure_prob]
                # # Generate failures and ensure at least one edge fails
                # while true
                #     failed_edges_k = [(i,j) for (i,j) in edges if rand() <= failure_prob]
                #     if !isempty(failed_edges_k)  # Only accept if we have at least one failure
                #         failed_edges[k] = failed_edges_k
                #         break
                #     end
                # end

                failed_edges[k] = [(i,j) for (i,j) in edges if rand() <= failure_prob]
                
                # Create scenario adjacency matrix (reuse memory)
                scenario_adj = copy(adj)
                for (i,j) in failed_edges[k]
                    scenario_adj[i,j] = false
                end
                
                # Check for perfect matching in scenario
                scenario_matching, _ = findmaxcardinalitybipartitematching(scenario_adj)
                if length(scenario_matching) != n
                    valid_scenarios = false
                    break
                end
            end
            
            if valid_scenarios
                # Find common failures across all scenarios
                always_failing = if !isempty(failed_edges)
                    common = Set(failed_edges[1])
                    for scenario in failed_edges[2:end]
                        common = intersect(common, Set(scenario))
                    end
                    common
                else
                    Set{Tuple{Int,Int}}()
                end
                
                # Update edges and failures
                updated_edges = filter(e -> e ∉ always_failing, edges)
                
                # No need to create new vectors, just filter in-place
                updated_failed_edges = [filter(e -> e ∉ always_failing, scenario) 
                                      for scenario in failed_edges]
                
                return updated_edges, updated_failed_edges
            end
        end
    end
    
    error("Could not generate valid graph and scenarios after $(max_graph_attempts) graph attempts")
end

"""
Save instance to JSON file
"""
function save_instance_to_json(filename::String,
                             n::Int,
                             edges::Vector{Tuple{Int,Int}},
                             num_scenarios::Int,
                             failed_edges::Vector{Vector{Tuple{Int,Int}}})
    # Convert data to JSON-compatible format
    data = Dict(
        "instance_info" => Dict(
            "n" => n,
            "num_edges" => length(edges),
            "num_scenarios" => num_scenarios
        ),
        "graph" => Dict(
            "edges" => [[e[1], e[2]] for e in edges]
        ),
        "scenarios" => Dict(
            "failed_edges" => [[[e[1], e[2]] for e in scenario] 
                             for scenario in failed_edges]
        )
    )
    
    # Create directory if it doesn't exist
    mkpath(dirname(filename))
    
    # Write to JSON file
    open(filename, "w") do f
        JSON.print(f, data, 2)  # 2 spaces for indentation
    end
end

"""
Generate instances based on configuration with hierarchical directory structure.
Parameters:
- testset: String specifying the test set ("grid", "expl", "large")
- base_dir: Optional override for base output directory

Returns number of instances generated
"""
function generate_instances(testset::String="grid", base_dir::Union{String,Nothing}=nothing)
    # Validate testset
    if !haskey(TESTSET_CONFIGS, testset)
        error("Unknown test set: $testset. Available options: $(join(keys(TESTSET_CONFIGS), ", "))")
    end
    
    # Get configuration for specified test set
    testset_config = TESTSET_CONFIGS[testset]
    output_dir = something(base_dir, testset_config.dir)
    
    total_instances = 0
    start_time = time()
    
    # Print generation settings
    println("\nGenerating $testset test set:")
    println("================================")
    println("Output directory: $output_dir")
    println("Instance sizes: $(length(testset_config.sizes)) configurations")
    println("Edge densities: $(testset_config.densities)")
    println("Failure probabilities: $(testset_config.probs)")
    println("Random seeds: $(testset_config.seeds)")
    println("================================\n")
    
    for (size_name, params) in testset_config.sizes
        size_start_time = time()
        instances_in_size = 0
        
        # Create size-specific directory
        size_dir = joinpath(output_dir, size_name)
        
        n = params["n"]
        num_scenarios = params["scenarios"]
       
        for density in testset_config.densities
            # Create density-specific directory
            density_str = "d$(density)"
            density_dir = joinpath(size_dir, density_str)
            
            for fail_prob in testset_config.probs
                # Create failure probability-specific directory
                fail_prob_str = "f$(fail_prob)"
                fail_prob_dir = joinpath(density_dir, fail_prob_str)
                mkpath(fail_prob_dir)
                
                for seed in testset_config.seeds
                    Random.seed!(seed)
                    
                    # Generate base graph and scenario failures
                    edges, failed_edges = generate_bipartite_with_scenarios(
                        n, density, num_scenarios, fail_prob)
                    
                    # Create filename
                    filename = joinpath(fail_prob_dir, 
                        "$(size_name)_d$(density)_f$(fail_prob)_se$(seed).json")
                    
                    # Save instance
                    save_instance_to_json(filename, n, edges, num_scenarios, failed_edges)
                    
                    total_instances += 1
                    instances_in_size += 1
                    
                    println("Generated instance: $(basename(filename)) in $(joinpath(size_name, density_str, fail_prob_str))")
                end
            end
        end
        
        # Print summary for this size category
        time_taken = round(time() - size_start_time, digits=2)
        println("\nCompleted $size_name instances:")
        println("  - Parameters: n=$(params["n"]), scenarios=$(params["scenarios"])")
        println("  - Generated: $instances_in_size instances")
        println("  - Time taken: $time_taken seconds")
    end
    
    # Print final summary
    total_time = round(time() - start_time, digits=2)
    println("\nGeneration Complete:")
    println("===================")
    println("Test set: $testset")
    println("Total instances: $total_instances")
    println("Total time: $total_time seconds")
    println("Average time per instance: $(round(total_time/total_instances, digits=2)) seconds")
    
    return total_instances
end

"""
Generate uniform bipartite graph where each scenario has exactly one edge failure
Returns:
- edges: Vector of edges in the final graph 
- failed_edges: Vector of vectors containing scenario failures (one edge per scenario)
"""
function generate_uniform_bipartite_with_scenarios(n::Int,
                                                 density::Float64, 
                                                 num_scenarios::Int;
                                                 max_graph_attempts::Int=1000000,
                                                 max_scenario_attempts::Int=100)
    @assert 0 < density <= 1 "Density must be between 0 and 1"
    
    for graph_attempt in 1:max_graph_attempts
        # Generate initial edges
        edges = [(i,j) for i in 1:n for j in 1:n if rand() <= density]
        
        # Create adjacency matrix using BitArray for efficiency
        adj = falses(n, n)
        [adj[i,j] = true for (i,j) in edges]
        
        # Check if original graph has a perfect matching
        matching, matched = findmaxcardinalitybipartitematching(adj)
        if length(matching) != n
            continue  # Try another graph
        end

        # Try to generate valid scenarios
        for scenario_attempt in 1:max_scenario_attempts
            # Create scenarios where each one removes exactly one edge
            failed_edges = Vector{Vector{Tuple{Int,Int}}}(undef, num_scenarios)
            valid_scenarios = true
            

            # Generate and test each scenario
            for k in 1:num_scenarios
                # Select a random edge to fail
                failed_edge = rand(collect(Set(edges)))
                failed_edges[k] = [failed_edge]
                
                # Create scenario adjacency matrix
                scenario_adj = copy(adj)
                scenario_adj[failed_edge[1], failed_edge[2]] = false
                
                # Check for perfect matching in scenario
                scenario_matching, _ = findmaxcardinalitybipartitematching(scenario_adj)
                if length(scenario_matching) != n
                    valid_scenarios = false
                    break
                end
            end
            
            if valid_scenarios
                return edges, failed_edges
            end
        end
    end
    
    error("Could not generate valid graph and scenarios after $(max_graph_attempts) graph attempts")
end

"""
Generate uniform instances based on configuration with hierarchical directory structure
"""
function generate_uniform_instances(testset::String="suni", base_dir::Union{String,Nothing}=nothing)
    # Validate testset
    if !haskey(TESTSET_CONFIGS, testset)
        error("Unknown test set: $testset. Available options: $(join(keys(TESTSET_CONFIGS), ", "))")
    end
    
    # Get configuration for specified test set
    testset_config = TESTSET_CONFIGS[testset]
    output_dir = something(base_dir, testset_config.dir)
    
    total_instances = 0
    start_time = time()
    
    # Print generation settings
    println("\nGenerating $testset test set:")
    println("================================")
    println("Output directory: $output_dir")
    println("Instance sizes: $(length(testset_config.sizes)) configurations")
    println("Edge densities: $(testset_config.densities)")
    println("Random seeds: $(testset_config.seeds)")
    println("================================\n")
    
    for (size_name, params) in testset_config.sizes
        size_start_time = time()
        instances_in_size = 0
        
        # Create size-specific directory
        size_dir = joinpath(output_dir, size_name)
        
        n = params["n"]
        num_scenarios = params["scenarios"]
       
        for density in testset_config.densities
            # Create density-specific directory
            density_str = "d$(density)"
            density_dir = joinpath(size_dir, density_str)
            mkpath(density_dir)
            
            for fail_prob in testset_config.probs
                # Create failure probability-specific directory
                fail_prob_str = "f$(fail_prob)"
                fail_prob_dir = joinpath(density_dir, fail_prob_str)
                mkpath(fail_prob_dir)
                for seed in testset_config.seeds
                    Random.seed!(seed)
                    
                    # Generate base graph and uniform scenario failures
                    edges, failed_edges = generate_uniform_bipartite_with_scenarios(
                        n, density, num_scenarios)
                    
                    # Create filename
                    filename = joinpath(fail_prob_dir, 
                        "$(size_name)_d$(density)_f$(fail_prob)_se$(seed).json")
                    
                    # Save instance
                    save_instance_to_json(filename, n, edges, num_scenarios, failed_edges)
                    
                    total_instances += 1
                    instances_in_size += 1
                    
                    println("Generated instance: $(basename(filename)) in $(joinpath(size_name, fail_prob_dir))")
                end
            end
        end
        
        # Print summary for this size category
        time_taken = round(time() - size_start_time, digits=2)
        println("\nCompleted $size_name instances:")
        println("  - Parameters: n=$(params["n"]), scenarios=$(params["scenarios"])")
        println("  - Generated: $instances_in_size instances")
        println("  - Time taken: $time_taken seconds")
    end
    
    # Print final summary
    total_time = round(time() - start_time, digits=2)
    println("\nGeneration Complete:")
    println("===================")
    println("Test set: $testset")
    println("Total instances: $total_instances")
    println("Total time: $total_time seconds")
    println("Average time per instance: $(round(total_time/total_instances, digits=2)) seconds")
    
    return total_instances
end

"""
Generate bipartite graph with scenarios designed for presolve testing.
Creates a structured set of scenarios with guaranteed dominance relationships.
"""
function generate_presolve_bipartite_with_scenarios(
    n::Int, 
    density::Float64,
    num_scenarios::Int,
    base_fail_prob::Float64;
    max_graph_attempts::Int=1000000,
    max_scenario_attempts::Int=100)

    for graph_attempt in 1:max_graph_attempts
        # Generate dense graph
        edges = [(i,j) for i in 1:n for j in 1:n if rand() <= density]
        
        # Create adjacency matrix for validation
        adj = falses(n, n)
        [adj[i,j] = true for (i,j) in edges]
        
        # Verify original graph has perfect matching
        matching, matched = findmaxcardinalitybipartitematching(adj)
        if length(matching) != n
            continue
        end

        # Calculate number of scenarios of each type
        num_core = round(Int, PRE_SCENARIO_CONFIG["core_ratio"] * num_scenarios)
        num_extended = round(Int, PRE_SCENARIO_CONFIG["extended_ratio"] * num_scenarios)
        num_random = num_scenarios - num_core - num_extended

        for scenario_attempt in 1:max_scenario_attempts
            try
                scenarios_failed_edges = Vector{Vector{Tuple{Int,Int}}}(undef, num_scenarios)
                valid_scenarios = true
                scenario_idx = 1

                # Helper function to check scenario validity
                function is_valid_scenario(failed_edges_k)
                    scenario_adj = copy(adj)
                    for (i,j) in failed_edges_k
                        scenario_adj[i,j] = false
                    end
                    scenario_matching, _ = findmaxcardinalitybipartitematching(scenario_adj)
                    return length(scenario_matching) == n
                end

                # Generate core scenarios (these will dominate others)
                for _ in 1:num_core
                    num_failures = max(1, round(Int, PRE_SCENARIO_CONFIG["core_failure_ratio"] * length(edges)))
                    edges_vec = collect(edges)
                    failed_indices = randperm(length(edges_vec))[1:num_failures]
                    failed_edges_k = edges_vec[failed_indices]
                    
                    if !is_valid_scenario(failed_edges_k)
                        valid_scenarios = false
                        break
                    end
                    scenarios_failed_edges[scenario_idx] = failed_edges_k
                    scenario_idx += 1
                end

                if !valid_scenarios
                    continue
                end

                # Generate extended scenarios (these will be dominated by core scenarios)
                for _ in 1:num_extended
                    # Pick a random core scenario and extend it
                    base_scenario = scenarios_failed_edges[rand(1:num_core)]
                    additional_failures = max(1, round(Int, PRE_SCENARIO_CONFIG["extended_failure_ratio"] * length(edges)))
                    remaining_edges = collect(setdiff(edges, base_scenario))
                    extra_indices = randperm(length(remaining_edges))[1:additional_failures]
                    extra_edges = remaining_edges[extra_indices]
                    failed_edges_k = vcat(base_scenario, extra_edges)
                    
                    if !is_valid_scenario(failed_edges_k)
                        valid_scenarios = false
                        break
                    end
                    scenarios_failed_edges[scenario_idx] = failed_edges_k
                    scenario_idx += 1
                end

                if !valid_scenarios
                    continue
                end

                # Generate random scenarios
                for _ in 1:num_random
                    failure_ratio = rand(range(PRE_SCENARIO_CONFIG["random_failure_range"]...))
                    num_failures = max(1, round(Int, failure_ratio * length(edges)))

                    edges_vec = collect(edges)
                    failed_indices = randperm(length(edges_vec))[1:num_failures]
                    failed_edges_k = edges_vec[failed_indices]
                    
                    if !is_valid_scenario(failed_edges_k)
                        valid_scenarios = false
                        break
                    end
                    scenarios_failed_edges[scenario_idx] = failed_edges_k
                    scenario_idx += 1
                end

                if valid_scenarios
                    # Scale failure sets based on base_fail_prob
                    if base_fail_prob < PRE_BASE_FAILURE_PROBS[1] # If using smaller probability
                        # Reduce size of failure sets proportionally
                        scale_factor = base_fail_prob / PRE_BASE_FAILURE_PROBS[1]
                        scenarios_failed_edges = [
                            # Use randperm for scaling too
                            let indices = randperm(length(scenario))[1:max(1, round(Int, length(scenario) * scale_factor))]
                                scenario[indices]
                            end
                            for scenario in scenarios_failed_edges
                        ]
                    end

                    return edges, scenarios_failed_edges
                end

            catch e
                println("Error in scenario generation: ", e)
                continue
            end
        end
    end
    
    error("Could not generate valid presolve instance after $(max_graph_attempts) attempts")
end

"""
Generate instances based on presolve-specific settings with hierarchical directory structure.
Generates instances with clear dominance relationships based on failed edges.
"""
function generate_presolve_instances(base_output_dir::String="data/pre")
    total_instances = 0
    start_time = time()
    
    println("\nGenerating Presolve Test Instances:")
    println("==================================")
    println("Output directory: $base_output_dir")
    println("Instance sizes: $(length(PRE_INSTANCE_SIZES)) configurations")
    println("Edge densities: $(PRE_EDGE_DENSITIES)")
    println("Base failure probabilities: $(PRE_BASE_FAILURE_PROBS)")
    println("Random seeds: $(PRE_RANDOM_SEEDS)")
    println("==================================\n")
    
    for (size_name, params) in PRE_INSTANCE_SIZES
        size_start_time = time()
        instances_in_size = 0
        
        n = params["n"]
        num_scenarios = params["scenarios"]
        
        # Create size-specific directory
        size_dir = joinpath(base_output_dir, size_name)
        
        for density in PRE_EDGE_DENSITIES
            # Create density-specific directory with proper formatting
            density_str = "d$(density)"
            density_dir = joinpath(size_dir, density_str)
            
            for base_fail_prob in PRE_BASE_FAILURE_PROBS
                # Create failure probability-specific directory with proper formatting
                fail_prob_str = "f$(base_fail_prob)"
                fail_prob_dir = joinpath(density_dir, fail_prob_str)
                mkpath(fail_prob_dir)
                
                for seed in PRE_RANDOM_SEEDS
                    Random.seed!(seed)
                    start_instance = time()
                    
                    # Generate base graph and failed edges using presolve-specific function
                    edges, failed_edges = generate_presolve_bipartite_with_scenarios(
                        n, density, num_scenarios, base_fail_prob)
                    
                    # Create instance filename
                    filename = joinpath(fail_prob_dir, 
                        "$(size_name)_$(density_str)_$(fail_prob_str)_se$(seed).json")
                    
                    # Create raw instance data with failed edges
                    raw_data = RawInstanceData(n, edges, num_scenarios, failed_edges)
                    
                    # Save instance
                    save_instance_to_json(filename, n, edges, num_scenarios, failed_edges)
                    
                    total_instances += 1
                    instances_in_size += 1
                    
                    instance_time = time() - start_instance
                    println("Generated instance: $(basename(filename)) ",
                           "in $(joinpath(size_name, density_str, fail_prob_str)) ",
                           "($(round(instance_time, digits=2))s)")

                    # Print dominance statistics for verification
                    if density == 1.0  # Only for complete graphs as they're most interesting
                        num_edges = length(edges)
                        core_failures = round(Int, mean([length(failed_edges[i]) 
                            for i in 1:round(Int, PRE_SCENARIO_CONFIG["core_ratio"] * num_scenarios)]))
                        extended_failures = round(Int, mean([length(failed_edges[i]) 
                            for i in (round(Int, PRE_SCENARIO_CONFIG["core_ratio"] * num_scenarios) + 1):
                                     (round(Int, (PRE_SCENARIO_CONFIG["core_ratio"] + 
                                                PRE_SCENARIO_CONFIG["extended_ratio"]) * num_scenarios))]))
                        println("  Failure statistics:")
                        println("    Total edges: $num_edges")
                        println("    Average core failures: $core_failures ($(round(100*core_failures/num_edges, digits=1))%)")
                        println("    Average extended failures: $extended_failures ($(round(100*extended_failures/num_edges, digits=1))%)")
                    end
                end
            end
        end
        
        # Print summary for this size category
        time_taken = round(time() - size_start_time, digits=2)
        println("\nCompleted $size_name instances:")
        println("  - Parameters: n=$(params["n"]), scenarios=$(params["scenarios"])")
        println("  - Generated: $instances_in_size instances")
        println("  - Time taken: $time_taken seconds")
    end
    
    # Print final summary
    total_time = round(time() - start_time, digits=2)
    println("\nGeneration Complete:")
    println("===================")
    println("Total instances: $total_instances")
    println("Total time: $total_time seconds")
    println("Average time per instance: $(round(total_time/total_instances, digits=2)) seconds")
    
    return total_instances
end
"""
Create a small test instance
"""
function create_test_instance(output_dir::String="data/raw")
    n = 4
    edges = [
        (1,1), (1,2), (1,3), (1,4),
        (2,1), (2,2), (2,3), (2,4),
        (3,1), (3,2), (3,3), (3,4),
        (4,1), (4,2), (4,3), (4,4)
    ]
    
    failed_edges = [
        [(1,2), (2,2), (2,3), (2,4), (3,1), (3,4), (4,1), (4,3), (4,4)],      # Scenario 1 failures
        [(1,1), (3,3)],      # Scenario 2 failures
        [(2,1), (4,1)]       # Scenario 3 failures
    ]
    
    filename = joinpath(output_dir, "test_instance.json")
    save_instance_to_json(filename, n, edges, length(failed_edges), failed_edges)
    println("Generated test instance: $(basename(filename))")
    return RawInstanceData(n, edges, length(failed_edges), failed_edges)
end

"""
Create a test instance specifically designed to test presolve
"""
function create_test_presolve_instance(output_dir::String="data/raw")
    n = 4
    edges = [
        (1,1), (1,2), (1,3), (1,4),
        (2,1), (2,2), (2,3), (2,4),
        (3,1), (3,2), (3,3), (3,4),
        (4,1), (4,2), (4,4)
    ]
    
    # Scenario 1: Almost complete graph (dominates many others)
    # Only missing (1,1) and (2,2)
    failed_edges_1 = [(1,1), (2,2)]
    
    # Scenario 2: Missing several edges (dominated by 1)
    # Missing (1,1), (2,2), (3,3), (4,4)
    failed_edges_2 = [(1,1), (2,2), (3,3)]
    
    # Scenario 3: Missing many edges (dominated by 1 and 2)
    # Missing (1,1), (2,2), (3,3), (4,4), (1,2), (2,1)
    failed_edges_3 = [(1,1), (2,2), (3,3), (4,4), (1,2), (2,1)]
    
    # Scenario 4: Very sparse (dominated by all others)
    # Only has edges (1,3), (1,4), (2,3), (2,4), (3,1), (3,2), (4,1), (4,2)
    failed_edges_4 = [(1,1), (1,2), (2,1), (2,2), (3,3), (3,4), (4,4)]
    
    # Scenario 5: Medium sparsity
    failed_edges_5 = [(1,1), (3,3), (1,2), (2,1)]
    
    failed_edges = [
        failed_edges_1,
        failed_edges_2,
        failed_edges_3,
        failed_edges_4,
        failed_edges_5
    ]
    
    filename = joinpath(output_dir, "presolve_test_instance.json")
    save_instance_to_json(filename, n, edges, length(failed_edges), failed_edges)
    println("Generated presolve test instance: $(basename(filename))")

    return RawInstanceData(n, edges, length(failed_edges), failed_edges)
end

"""
Create a test instance with duplicate scenarios
"""
function create_duplicate_test_instance(output_dir::String="data/raw")
    n = 4
    edges = [
        (1,1), (1,2), (1,3), (1,4),
        (2,1), (2,2), (2,3), (2,4),
        (3,1), (3,2), (3,3), (3,4),
        (4,1), (4,2), (4,3), (4,4)
    ]
    
    # Create 6 scenarios where:
    # - Scenarios 1, 2, and 3 are identical (each initially with prob 1/6)
    # - Scenarios 4 and 5 are identical (each initially with prob 1/6)
    # - Scenario 6 is unique (initially with prob 1/6)
    
    # After merging, we should have 3 unique scenarios with probabilities:
    # - Merged scenario (1,2,3) with prob 0.5
    # - Merged scenario (4,5) with prob 0.333...
    # - Unique scenario 6 with prob 0.166...
    
    failed_edges = [
        # Scenarios 1,2,3 (identical) - will merge to prob 0.5
        [(1,1), (2,2)],              # Scenario 1
        [(1,1), (2,2)],              # Scenario 2 (same as 1)
        [(1,1), (2,2)],              # Scenario 3 (same as 1)
        
        # Scenarios 4,5 (identical) - will merge to prob 0.333...
        [(3,3), (4,4)],              # Scenario 4
        [(3,3), (4,4)],              # Scenario 5 (same as 4)
        
        # Unique scenario - will keep prob 0.166...
        [(1,1), (2,2), (3,3)]        # Scenario 6
    ]
    
    filename = joinpath(output_dir, "duplicate_test_instance.json")
    mkpath(output_dir)
    save_instance_to_json(filename, n, edges, length(failed_edges), failed_edges)
    println("Generated duplicate test instance: $(basename(filename))")
    
    return RawInstanceData(n, edges, length(failed_edges), failed_edges)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Generate all instances
    generate_instances()
    
    # Generate test instance
    create_test_instance()
end