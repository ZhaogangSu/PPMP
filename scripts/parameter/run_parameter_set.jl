using JSON

include("../run_experiments.jl")
# Test configurations
TEST_CONFIGS = Dict(
    "base" => Dict(
        "name" => "BASE",
        "options" => [
            "--max-node", "1000000",
            "--is-root-restart", "false",
            "--is-tree-restart", "false",
            "--create-from-scratch", "false",
            "--cpx-presolve", "1",
            "--tree-user-cuts-per-node", "0",
            "--print-level", "0"
        ]
    ),
    "root_only" => Dict(
        "name" => "ROOT_ONLY",
        "options" => [
            "--max-node", "1000000",
            "--is-root-restart", "false",
            "--is-tree-restart", "false",
            "--create-from-scratch", "false",
            "--cpx-presolve", "1",
            "--tree-user-cuts-per-node", "0",
            "--print-level", "0",
            "--root-user-violation", "0.1",
            "--tree-user-cuts-per-node", "0"
        ]
    ),
    "full_tree" => Dict(
        "name" => "FULL_TREE",
        "options" => [
            "--max-node", "1000000",
            "--is-root-restart", "false",
            "--is-tree-restart", "false",
            "--create-from-scratch", "false",
            "--cpx-presolve", "1",
            "--tree-user-cuts-per-node", "5",
            "--print-level", "0",
            "--root-user-violation", "0.1",
            "--tree-user-violation", "0.1"
        ]
    ),
    "violation_test" => Dict(
        "name" => "VIOLATION",
        "options" => [
            "--max-node", "1000000",
            "--is-root-restart", "false",
            "--is-tree-restart", "false",
            "--create-from-scratch", "false",
            "--cpx-presolve", "1",
            "--tree-user-cuts-per-node", "5",
            "--print-level", "0",
            "--root-user-violation", "1e-5",
            "--tree-user-violation", "1e-5"
        ]
    )
)

"""
Run a specific test configuration
"""
function run_test_config(config_key::String, test_folders::Vector{String}, epsilon_values::Vector{Float64}, bin_file::String)
    config = TEST_CONFIGS[config_key]
    test_name = config["name"]
    
    # Create results directory
    result_dir = "results/raw/$(test_name)"
    mkpath(result_dir)
    
    # Submit jobs with the specified configuration
    lsf_command = submit_jobs("grid", test_folders, "FCS", epsilon_values, test_name, bin_file, 20)
    
    return lsf_command
end

"""
Process and analyze results for a specific test configuration
"""
function process_test_results(config_key::String)
    config = TEST_CONFIGS[config_key]
    test_name = config["name"]
    
    # Collect and process results
    results = collect_results("results/raw/$(test_name)", "Grid")
    df = results_to_dataframe(results)
    
    # Create output directories
    mkpath("processed/$(test_name)")
    mkpath("figures/$(test_name)")
    
    # Generate statistics and visualizations
    stats = generate_summary_stats(df, save_path="processed/$(test_name)/statistics.csv")
    visualize_solving_times(df, save_path="figures/$(test_name)")
    
    return df, stats
end

"""
Run all experiments and process results
"""
function run_all_experiments(test_folders::Vector{String}, epsilon_values::Vector{Float64}, bin_file::String)
    results = Dict()
    commands = Dict()
    
    # Run each test configuration
    for config_key in keys(TEST_CONFIGS)
        println("\nRunning test configuration: $(config_key)")
        command = run_test_config(config_key, test_folders, epsilon_values, bin_file)
        commands[config_key] = command
    end
    
    return commands
end

"""
Process results for all experiments
"""
function process_all_results()
    all_results = Dict()
    
    for config_key in keys(TEST_CONFIGS)
        println("\nProcessing results for configuration: $(config_key)")
        df, stats = process_test_results(config_key)
        all_results[config_key] = (df=df, stats=stats)
    end
    
    return all_results
end

# Example usage:
# test_folders = ["test_20_16"]
# epsilon_values = [0.05, 0.1, 0.2]
# bin_file = "/path/to/binary"

# # Run experiments
# commands = run_all_experiments(test_folders, epsilon_values, bin_file)

# # Process results later
# results = process_all_results()