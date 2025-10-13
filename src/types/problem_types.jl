# src/types/problem_types.jl



"""
Raw instance data structure
"""
struct RawInstanceData
    n::Int                           # Nodes on each side
    edges::Vector{Tuple{Int,Int}}    # Original graph edges
    num_scenarios::Int               # Number of scenarios
    failed_edges::Vector{Vector{Tuple{Int,Int}}}  # Failed edges per scenario
end

"""
Represents a PPMP instance
"""
struct PPMPInstance
    n::Int                           # Number of nodes on each side
    edges::Vector{Tuple{Int,Int}}    # Original graph edges
    scenarios::Vector{Vector{Tuple{Int,Int}}}  # Available edges in each scenario
    failed_edges::Vector{Vector{Tuple{Int,Int}}}  # Failed edges in each scenario
    probabilities::Vector{Float64}   # Scenario probabilities
    costs::Vector{Float64}           # Edge costs
    epsilon::Float64                 # Risk tolerance
    edge_to_index::Dict{Tuple{Int,Int}, Int}  # Maps edge to its index
    
    # Constructor from raw data
    function PPMPInstance(raw_data::RawInstanceData,
                         probabilities::Vector{Float64},
                         costs::Vector{Float64},
                         epsilon::Float64)
        edge_to_index = Dict(e => i for (i, e) in enumerate(raw_data.edges))
        
        # Store both scenarios and failed edges
        scenarios = Vector{Vector{Tuple{Int,Int}}}(undef, raw_data.num_scenarios)
        failed_edges = raw_data.failed_edges  # Store failed edges directly
        
        # Construct scenario edge sets from failed edges
        for k in 1:raw_data.num_scenarios
            # Scenario k contains all edges except the failed ones
            scenarios[k] = filter(e -> !(e in failed_edges[k]), raw_data.edges)
        end
        
        # Validation
        @assert length(probabilities) == raw_data.num_scenarios "Mismatch in number of scenarios"
        @assert isapprox(sum(probabilities), 1.0, atol=1e-6) "Probabilities must sum to 1"
        @assert 0 <= epsilon <= 1 "Epsilon must be between 0 and 1"
        @assert length(costs) == length(raw_data.edges) "Mismatch between edges and costs"
        
        new(raw_data.n, raw_data.edges, scenarios, failed_edges, probabilities, 
            costs, epsilon, edge_to_index)
    end

    # Constructor for reduced instance
    function PPMPInstance(original::PPMPInstance,
                         kept_scenarios::Union{Vector{Int}, AbstractSet{Int}},
                         new_epsilon::Float64)
        num_kept = length(kept_scenarios)
        
        # Keep both scenarios and failed edges for kept scenarios
        new_scenarios = Vector{Vector{Tuple{Int,Int}}}(undef, num_kept)
        new_failed_edges = Vector{Vector{Tuple{Int,Int}}}(undef, num_kept)
        new_probabilities = Vector{Float64}(undef, num_kept)
        
        for (new_idx, old_idx) in enumerate(kept_scenarios)
            new_scenarios[new_idx] = original.scenarios[old_idx]
            new_failed_edges[new_idx] = original.failed_edges[old_idx]
            new_probabilities[new_idx] = original.probabilities[old_idx]
        end
        
        new(original.n, 
            original.edges,
            new_scenarios,
            new_failed_edges,
            new_probabilities,
            original.costs,
            new_epsilon,
            original.edge_to_index)
    end
end

# Helper function to create test instance with default parameters
function create_test_ppmp_instance()
    raw_data = create_test_instance()
    probs = [0.5, 0.3, 0.2]
    costs = ones(length(raw_data.edges))  # Simple unit costs
    epsilon = 0.2

    return PPMPInstance(raw_data, probs, costs, epsilon)
end

# Helper function to create test instance with default parameters
function create_test_ppmp_presolve_instance()
    raw_data = create_test_presolve_instance()
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    costs = ones(length(raw_data.edges))  # Simple unit costs
    epsilon = 0.2

    return PPMPInstance(raw_data, probs, costs, epsilon)
end