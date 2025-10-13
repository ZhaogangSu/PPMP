# src/types/callback_types.jl


"""
Store both lazy and user cuts for restart
"""
mutable struct StoredCut
    coefficients::Vector{Float64}
    rhs::Float64
    fingerprint::String
    violation::Float64
    norm::Float64
    is_lazy::Bool  # true for lazy cuts, false for user cuts
    in_round::Int       # Track at which round of solving get this cut (if duplicate, keep the latest)
    depth::Int          # Track at which depth of the tree get this cut (if duplicate, keep the latest)

    StoredCut(coefficients::Vector{Float64}, 
            rhs::Float64, 
            fingerprint::String,
            violation::Float64, 
            norm::Float64, 
            is_lazy::Bool, 
            in_round::Int, 
            depth::Int
        ) = new(coefficients, rhs, fingerprint, violation, norm, is_lazy, in_round, depth)
end

"""
Cut pool management structure
"""
mutable struct CutPool
    fingerprints::Set{String}                  # For dupilicity check
    fingerprint_to_index::Dict{String,Int}     # For fast index lookup
    cuts::Vector{StoredCut}                    # Stored cuts
    fingerprint_hit::Int                      # Track how many fingerprints in the pool were hit (duplicated) during callbacks
    pool_size::Int
    
    
    CutPool() = new(Set{String}(), Dict{String,Int}(), Vector{StoredCut}(), 0, 0)
end 


"""
Callback data structure for flow cut generation
"""
mutable struct FlowCutGlbPool
    lazy_pool::CutPool            # Cut pool management
    user_pool::CutPool            # Cut pool management

    # Updated at user cut callback
    root_fraction_sol::Vector{Float64}  # Fractional solution at the root node
    root_fraction_obj::Float64         # Objective value at the root node
    best_fraction_sol::Vector{Float64}  # Best fractional solution at the root node
    best_fraction_obj::Float64         # Best fractional objective value at the root node


    FlowCutGlbPool() = new(CutPool(), CutPool(), Float64[], 0.0, Float64[], 0.0)
end


"""
Callback statistics structure for tracking performance
"""
mutable struct CallbackStats
    # Overall statistics and Separate user/lazy statistics
    total_cuts_added::Int
    total_callbacks::Int

    user_callbacks::Int
    lazy_callbacks::Int

    total_callbacks_time::Float64
    user_callbacks_time::Float64
    lazy_callbacks_time::Float64
    
    # Cut tracking
    cur_node_idx::Int
    user_cuts_cur_node::Int
    lazy_cuts_cur_node::Int

    # Round tracking
    user_rounds_cur_node::Int
    lazy_rounds_cur_node::Int
    
    # Cut statistics
    cuts_per_scenario::Dict{Int,Int}
    # cuts_per_iteration::Vector{Int}
    cut_violations::Vector{Float64}

    # Mixing inequalities statistics
    lazy_mixing_cuts_added::Int              # Total lazy mixing cuts added
    user_mixing_cuts_added::Int              # Total user mixing cuts added

    lazy_mixing_cuts_time::Float64           # Total time spent on lazy mixing cuts
    user_mixing_cuts_time::Float64           # Total time spent on user mixing cuts
    
    # Root node statistics
    root_stats::Dict{String,Any}
    
    # Tree node statistics
    tree_stats::Dict{String,Any}

    # Restart statistics
    is_restart::Bool
    num_stored_constraints::Int
    stored_constraints_violated::Int  # Track how many stored constraints were actually needed

    num_lazy_cons_in_pool::Int # global lazy cons pool's size at this round
    num_user_cuts_in_pool::Int # global user cuts pool's size at this round

    num_lazy_cons_to_add::Int # number of lazy cons added at this solving round
    num_user_cuts_to_add::Int # number of user cuts added at this solving round

    num_lazy_active::Int # number of lazy cons active at this solving round
    num_user_active::Int # number of user cuts active at this solving round

    lazy_fingerprint_hit::Int # Track how many fingerprints in the lazy pool were hit (duplicated) during callbacks
    user_fingerprint_hit::Int # Track how many fingerprints in the user pool were hit (duplicated) during callbacks


    CallbackStats() = new(
        0, 0, 0, 0,  # Overall
        0.0, 0.0, 0.0,       # User/Lazy counts
        0, 0, 0,  # Cut tracking
        0, 0,      # Round tracking
        Dict{Int,Int}(),  # Cuts per scenario
        Float64[],  # Cut violations

        0,                  # lazy_mixing_cuts_added
        0,                  # user_mixing_cuts_added

        0.0,               # lazy_mixing_cuts_time
        0.0,               # user_mixing_cuts_time

        Dict{String,Any}(  # Root stats
            "user_cuts_added" => 0,
            "lazy_cuts_added" => 0,
            "user_mixing_cuts_added" => 0,
            "lazy_mixing_cuts_added" => 0,
            "max_violation" => 0.0,
            "mixing_max_violation" => 0.0
        ),
        Dict{String,Any}(  # Tree stats
            "user_cuts_added" => 0,
            "lazy_cuts_added" => 0,
            "user_mixing_cuts_added" => 0,
            "lazy_mixing_cuts_added" => 0,
            "nodes_processed" => 0
        ),
        false,          # is_restart
        0,             # num_stored_constraints
        0,              # stored_constraints_violated
        0,              # num_lazy_cons_in_pool
        0,              # num_user_cuts_in_pool
        0,             # num_lazy_cons_to_add
        0,              # num_user_cuts_to_add
        0,              # num_lazy_active
        0,              # num_user_active
        0,              # lazy_fingerprint_hit
        0               # user_fingerprint_hit
    )
end