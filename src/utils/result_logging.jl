# src/utils/result_logging.jl

using JSON
using Dates

"""
Store results from a solver run
"""
struct ResultData
    instance_name::String
    solver_name::String
    solve_time::Float64
    setup_time::Float64
    node_count::Int
    obj_value::Float64
    obj_bound::Float64
    gap::Float64
    status::String
    selected_edges::Vector{Int}
    scenario_values::Vector{Bool}
    timestamp::String
    callback_stats::Union{Nothing,CallbackStats}
    merge_stats::Union{Nothing,MergeStats}          # Merge statistics
    presolve_stats::Union{Nothing,PresolveStats}    # Presolve statistics
end

"""
Convert solution to result data
"""
function create_result_data(solution::PPMPSolution,
                          instance_name::String,
                          solver_name::String;
                          merge_stats::Union{Nothing,MergeStats}=nothing,
                          presolve_stats::Union{Nothing,PresolveStats}=nothing
                          )
    return ResultData(
        instance_name,
        solver_name,
        solution.stats.solve_time,
        solution.stats.setup_time,
        solution.stats.node_count,
        solution.stats.obj_value,
        solution.stats.obj_bound,
        solution.stats.gap,
        string(solution.stats.status),
        Vector{Int}(),
        # solution.selected_edges,
        solution.scenario_values,
        Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"),
        solution.stats.callback_stats,
        merge_stats,
        presolve_stats
    )
end

"""
Save results to JSON
"""
function save_results(result::ResultData, output_dir::String)
    # Create results directory if it doesn't exist
    mkpath(output_dir)
    
    # Determine solver suffix
    suffix = if result.solver_name == "OriginalMIPSolver"    "_OMS"
             elseif result.solver_name == "FlowCutSolver"    "_FCS"
             else
                 throw(ArgumentError("Unsupported solver type: $(result.solver_name)"))
             end
    
    # Create result filename with suffix
    filename = joinpath(output_dir, "$(result.instance_name)$(suffix).json")
    
    result_dict = Dict(
        "instance_name" => result.instance_name,
        "solver_name" => result.solver_name,
        "solve_time" => result.solve_time,
        "setup_time" => result.setup_time,
        "node_count" => result.node_count,
        "obj_value" => result.obj_value,
        "obj_bound" => result.obj_bound,
        "gap" => result.gap,
        "status" => result.status,
        "selected_edges" => result.selected_edges,
        "scenario_values" => result.scenario_values,
        "timestamp" => result.timestamp
    )
    
    # Add callback statistics if available
    if !isnothing(result.callback_stats)
        stats = result.callback_stats
        result_dict["callback_statistics"] = Dict(
            "total_cuts_added" => stats.total_cuts_added,
            "total_callbacks" => stats.total_callbacks,
            "cuts_per_scenario" => stats.cuts_per_scenario,
            "cut_violations" => Dict(
                "max" => isempty(stats.cut_violations) ? 0.0 : maximum(stats.cut_violations),
                "min" => isempty(stats.cut_violations) ? 0.0 : minimum(stats.cut_violations),
                "mean" => isempty(stats.cut_violations) ? 0.0 : mean(stats.cut_violations),
                "median" => isempty(stats.cut_violations) ? 0.0 : median(stats.cut_violations)
            )
        )
    end

    # Add merge statistics if available
    if !isnothing(result.merge_stats)
        stats = result.merge_stats
        result_dict["merge_statistics"] = Dict(
            "total_scenarios" => stats.total_scenarios,
            "unique_scenarios" => stats.num_unique_scenarios,
            "duplicated_scenarios" => stats.num_duplicated,
            "group_sizes" => Dict(
                "max" => stats.max_group_size,
                "min" => stats.min_group_size,
                "avg" => stats.avg_group_size,
                "median" => stats.median_group_size
            ),
            "scenario_groups" => [Dict(
                "indices" => group[1],
                "size" => group[2],
                "probability" => group[3]
            ) for group in stats.scenario_groups],
            "merge_time" => stats.merge_time
        )
    end

    return filename, result_dict
end