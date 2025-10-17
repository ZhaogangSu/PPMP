# scripts/run_experiments.jl

using PPMP
using Glob
using JSON
using Dates


"""
Get the number of cores per node by parsing LSF bhosts output
"""
function get_cores_per_node()
    try

        # Run bhosts command to get host information
        bhosts_output = read(`bhosts -l`, String)
        
        # Look for "MAXIMUM NJOBS" line which indicates cores per node
        for line in split(bhosts_output, '\n')
            if occursin("MAXIMUM NJOBS", line)
                # Extract the number after "MAXIMUM NJOBS"
                m = match(r"MAXIMUM NJOBS\s*:\s*(\d+)", line)
                if !isnothing(m)
                    return parse(Int, m[1])
                end
            end
        end        
        
        
        # Fallback if can't parse bhosts output
        @warn "Could not detect cores per node from bhosts, using lscpu as backup"

        # Try using lscpu as backup
        lscpu_output = read(`lscpu`, String)
        for line in split(lscpu_output, '\n')
            if occursin("CPU(s):", line)
                m = match(r"CPU\(s\):\s*(\d+)", line)
                if !isnothing(m)
                    return parse(Int, m[1])
                end
            end
        end

        
        error("Could not detect number of cores per node")
    catch e
        @warn "Error detecting cores per node: $e"
        @warn "Defaulting to 36 cores per node"
        return 36  # Default fallback value
    end
end

function create_job_script(instance_paths::Vector{String}, result_paths::Vector{String}, folders::Vector{String}, solver_type::String, epsilon_values::Vector{Float64}, bin_file::String, config_command::Vector{String}=String[])
    # Generate script name with metadata
    timestamp = Dates.format(now(), "mmdd_HHMM")
    num_instances = length(instance_paths)
    clean_folders = [replace(folder, "*" => "_") for folder in folders]
    folders_str = join(clean_folders, "-")    
    # Add random seed (using 4 digits for readability)
    random_seed = lpad(rand(1:999999), 6, '0')
    script_name = "run_$(timestamp)_$(folders_str)_i$(num_instances)_t1_s$(random_seed).sh"
    
    if solver_type == "FCS"
        solver = "FlowCutSolver"
    elseif solver_type == "OMS"
        solver = "OriginalMIPSolver"
    else
        error("Invalid solver type: $solver_type")
    end


    # Build the PPMP command with additional configuration
    config_str = isempty(config_command) ? "" : " " * join(["\"$arg\"" for arg in config_command], " ")

    script = """
    #!/bin/bash

    # PPMP Experiment Script
    # Generated: $(timestamp)
    # Instances: $(num_instances)
    # Threads: $(default_config().threads)
    # Time Limit: $(default_config().max_time) seconds
    # MIP Gap: $(default_config().mip_gap)
    
    """
    
    for (instance_path, result_path) in zip(instance_paths, result_paths)
        # Extract instance name components
        instance_filename = basename(instance_path)
        base_name = replace(instance_filename, ".json" => "")
        
        for epsilon in epsilon_values
            # Create matching names for log and result files
            file_base_name = "$(base_name)_eps$(epsilon)"
            log_file = joinpath(result_path, "$(file_base_name)")
            
            # Create result directory if it doesn't exist
            mkpath(dirname(log_file))

            # $(bin_file)/bin/PPMP $(solver) $epsilon "$(instance_path)" "$(result_path)" "--create-right-cons" "false" "--create-random-cons" "false" 2>&1 > "$(log_file)_$(solver_type).log"
            script *= """
            $(bin_file)/bin/PPMP $(solver) $epsilon "$(instance_path)" "$(result_path)"$(config_str) 2>&1 > "$(log_file)_$(solver_type).log"
            """
        end
    end
    
    # script *= "wait\n"
    
    # Create scripts directory if it doesn't exist
    mkpath("scripts/run_jobs")
    
    script_path = joinpath("scripts/run_jobs", script_name)
    write(script_path, script)
    chmod(script_path, 0o755)
    
    return script_path
end

"""
Get instance paths and create corresponding result directories based on test set
"""
function get_instance_paths(test_set::String, folders::Vector{String}, marker::String="")
    if !haskey(TESTSET_CONFIGS, test_set)
        error("Unknown test set: $test_set. Available options: $(join(keys(TESTSET_CONFIGS), ", "))")
    end
    
    testset_config = TESTSET_CONFIGS[test_set]
    instance_paths = String[]
    result_paths = String[]
    
    # Base result directory
    if !isempty(marker)
        # Extract the first part before underscore and convert to lowercase
        marker_parts = split(marker, '_')
        base_folder = lowercase(marker_parts[1])
        # Construct the full path: results/base_folder/marker
        result_base = joinpath("results", base_folder, marker)
    else
        result_base = joinpath("results", "raw")
    end

    result_based_exists = isdir(result_base)
    if result_based_exists
        @warn "Result directory $result_base already exists."
    end
    
    for folder in folders
        # Use glob pattern matching
        folder_pattern = joinpath("data", test_set, folder)
        matching_folders = glob(folder_pattern)
        
        for matched_folder in matching_folders
            folder_name = basename(matched_folder)
            result_folder = joinpath(result_base, folder_name)

            # Use densities from the corresponding test set configuration
            for density in testset_config.densities
                density_dir = joinpath(matched_folder, "d$(density)")
                result_density = joinpath(result_folder, "d$(density)")
                
                # Use failure probabilities from the corresponding test set configuration
                for fail_prob in testset_config.probs
                    fail_dir = joinpath(density_dir, "f$(fail_prob)")
                    result_fail = joinpath(result_density, "f$(fail_prob)")
                    
                    # Collect instance paths and corresponding result paths
                    if isdir(fail_dir)
                        for file in readdir(fail_dir)
                            if endswith(file, ".json")
                                instance_path = joinpath(fail_dir, file)
                                result_path = joinpath(result_fail, replace(file, ".json" => ""))
                                push!(instance_paths, instance_path)
                                push!(result_paths, result_path)
                                if !isdir(result_path)
                                    mkpath(result_path)
                                end    
                            end
                        end
                    end
                end
            end
        end
    end
    
    if !isempty(instance_paths)
        println("\nFound $(length(instance_paths)) instances")
        if !result_based_exists
            println("Created result directories in: $result_base")
        else 
            println("Result directories already exist")
        end
    else
        println("\nNo instances found in the specified folders")
    end
    
    return instance_paths, result_paths
end

"""
Submit jobs to LSF with test set configuration
"""
function submit_jobs(test_set::String, 
                    folders::Vector{String}, 
                    solver_type::String, 
                    epsilon_values::Vector{Float64}, 
                    testname::String, 
                    bin_file::String, 
                    num_node_sets::Int,
                    config_command::Vector{String}=String[])
    
    # Validate test set
    if !haskey(TESTSET_CONFIGS, test_set)
        error("Unknown test set: $test_set. Available options: $(join(keys(TESTSET_CONFIGS), ", "))")
        return
    end

    threads = 1
    cores_per_node = get_cores_per_node()
    
    # Adjust cores per node based on cluster
    if cores_per_node == 52
        num_node_sets = min(num_node_sets, 4)  # Small cluster limit
    elseif cores_per_node == 72
        cores_per_node = 36
    end
    
    # Get instance paths using test set configuration
    instance_paths, result_paths = get_instance_paths(test_set, folders, testname)
    
    if isempty(instance_paths)
        error("No instances found for test set $test_set in folders: $(join(folders, ", "))")
    end
    
    # Create job script
    script_path = create_job_script(instance_paths, result_paths, folders, solver_type, epsilon_values, bin_file, config_command)
    
    # Calculate total cores
    total_cores = cores_per_node * num_node_sets

    # Submit to LSF
    job_name = split(testname, "_")
    job_name = job_name[end]
    lsf_command = `bsub -J $(solver_type)_$(job_name)
                        -n $total_cores 
                        -o %J.out 
                        -e %J.err 
                        -q batch 
                        -R "span[ptile=$cores_per_node]" 
                        "mpirun ~/Mip/lsf/lsf $script_path $threads $cores_per_node"`
    
    println("\nSubmitting job with configuration:")
    println("===================================")
    println("Test set: $test_set")
    println("Folders: $(join(folders, ", "))")
    println("Solver: $solver_type")
    println("Epsilon values: $epsilon_values")
    println("Total cores: $total_cores ($num_node_sets nodes × $cores_per_node cores)")
    println("Script path: $script_path")
    if !isempty(config_command)
        println("Additional config: $(join(config_command, " "))")
    end
    println("===================================\n")
    run(lsf_command)
    return lsf_command
end


            
# Add command to script
# script *= """
# julia --sysimage=lib/PPMP.so --project -e 'using PPMP; run_instance(OriginalMIPSolver, $epsilon, "$(instance_path)", "$(result_path)")' 2>&1 > "$(log_file)_OMS.log
# """

# # If running directly
# if abspath(PROGRAM_FILE) == @__FILE__
#     submit_jobs()
# end 
# print("submit_jobs([\"test*\"], \"FCS\", [0.05, 0.1, 0.2], \"TEST_1\", num_node_sets=20)")
# println("submit_jobs(\"grid\", [\"test*\"], \"FCS\", [0.0, 0.05, 0.1, 0.2], \"Grid_FCS_ver4\", \"bin\", 40)")
# println("submit_jobs(\"dev\", [\"test*\"], \"FCS\", [0.0, 0.05, 0.1, 0.2], \"Dev_FCS_No_Init_ver1\", \"bin\", 20)")
# println("submit_jobs(\"dev\", [\"test*\"], \"FCS\", [0.0, 0.05, 0.1, 0.2], \"Dev_FCS_Left_Init_ver1\", \"bin\", 20)")
# println("submit_jobs(\"dev\", [\"test*\"], \"FCS\", [0.0, 0.05, 0.1, 0.2], \"Dev_FCS_LeftRight_InitCuts_ver1\", \"bin\", 20)")
# println("submit_jobs(\"dev\", [\"test*\"], \"FCS\", [0.0, 0.05, 0.1, 0.2], \"Dev_FCS_ver1\", \"bin\", 20)")


"""Suni

# Setting -1: No Auto 

config_1 = ["--auto-benders", "false"]
submit_jobs("suni", ["test*"], "OMS", [0.0, 0.01, 0.05, 0.1, 0.2], "Suni_OMS_no_auto_ver2", "bin", 15, config_1)

# Setting 0: Auto 

submit_jobs("suni", ["test*"], "OMS", [0.0, 0.01, 0.05, 0.1, 0.2], "Suni_OMS_ver2", "bin", 10)

# Setting 1: Default settings

submit_jobs("suni", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Suni_FCS_ver2", "bin", 5)

# Setting 2: No presolve

config2 = ["--is-presolve-flowcut", "false"]
submit_jobs("suni", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Suni_FCS_no_presolve_ver2", "bin", 5, config2)

# Setting 3: Disable mixing and presolve

config3 = ["--user-mixing-cuts-enabled", "false", "--lazy-mixing-cuts-enabled", "false", "--is-presolve-flowcut", "false"]
submit_jobs("suni", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Suni_FCS_no_mix_presolve_ver2", "bin", 5, config3)

"""

"""Dev
# Setting -1: No Auto 
config_1 = ["--auto-benders", "false"]
submit_jobs("dev", ["test*"], "OMS", [0.0, 0.01, 0.05, 0.1, 0.2], "Dev_OMS_no_auto_ver2", "bin", 15, config_1)


# Setting 0: Auto
submit_jobs("dev", ["test*"], "OMS", [0.0, 0.01, 0.05, 0.1, 0.2], "Dev_OMS_ver2", "bin", 10)

# Setting 1: Default settings
submit_jobs("dev", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Dev_FCS_ver2", "bin", 5)

# Setting 2: No presolve
config2 = ["--is-presolve-flowcut", "false"]
submit_jobs("dev", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Dev_FCS_no_presolve_ver2", "bin", 5, config2)


# Setting 3: Disable mixing and presolve
config3 = ["--user-mixing-cuts-enabled", "false", "--lazy-mixing-cuts-enabled", "false", "--is-presolve-flowcut", "false"]
submit_jobs("dev", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Dev_FCS_no_mix_presolve_ver2", "bin", 5, config3)

"""

"""Pre
# Setting -1: No Auto 
config_1 = ["--auto-benders", "false"]
submit_jobs("pre", ["test*"], "OMS", [0.0, 0.01, 0.05, 0.1, 0.2], "Pre_OMS_no_auto_ver2", "bin", 15, config_1)

# Setting 0: Auto
submit_jobs("pre", ["test*"], "OMS", [0.0, 0.01, 0.05, 0.1, 0.2], "Pre_OMS_ver2", "bin", 10)

# Setting 1: Default settings
submit_jobs("pre", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Pre_FCS_ver2", "bin", 5)

# Setting 2: No presolve
config2 = ["--is-presolve-flowcut", "false"]
submit_jobs("pre", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Pre_FCS_no_presolve_ver2", "bin", 5, config2)

# Setting 3: Disable mixing and presolve
config3 = ["--user-mixing-cuts-enabled", "false", "--lazy-mixing-cuts-enabled", "false", "--is-presolve-flowcut", "false"]
submit_jobs("pre", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], "Pre_FCS_no_mix_presolve_ver2", "bin", 5, config3)

"""


"""


# === Configuration Definitions ===

# Common parameters
epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.2]
instance_pattern = ["test*"]
binary_type = "bin"

# Solver configurations with time limit
config_no_auto = ["--auto-benders", "false", "--max-time", "3600"]
config_no_presolve_mixing = ["--user-mixing-cuts-enabled", "false", "--is-presolve-flowcut", "false", "--max-time", "3600"]
config_no_presolve = ["--is-presolve-flowcut", "false", "--max-time", "3600"]
config_default = ["--max-time", "3600"]



# === Suni Dataset ===
# Setting 0: OMS No Auto
submit_jobs("suni", instance_pattern, "OMS", epsilon_values, "Suni_OMS_no_auto_ver2", binary_type, 15, config_no_auto)

# Setting 1: OMS Auto
submit_jobs("suni", instance_pattern, "OMS", epsilon_values, "Suni_OMS_auto_ver2", binary_type, 10, config_default)

# Setting 2: FCS No Presolve & No Mixing
submit_jobs("suni", instance_pattern, "FCS", epsilon_values, "Suni_FCS_no_presolve_mixing_ver2", binary_type, 5, config_no_presolve_mixing)

# Setting 3: FCS No Presolve
submit_jobs("suni", instance_pattern, "FCS", epsilon_values, "Suni_FCS_no_presolve_ver3", binary_type, 20, config_no_presolve)

# Setting 4: FCS Default
submit_jobs("suni", instance_pattern, "FCS", epsilon_values, "Suni_FCS_default_ver3", binary_type, 20, config_default)



# Common parameters
epsilon_values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
instance_pattern = ["test_100*"]
binary_type = "bin_mix"
config_no_presolve = ["--is-presolve-flowcut", "false", "--max-time", "3600"]
config_default = ["--max-time", "3600"]

# Setting 3: FCS No Presolve
submit_jobs("pre", instance_pattern, "FCS", epsilon_values, "Pre5_fin_sparse_no_pre_2", binary_type, 20, config_no_presolve)

# Setting 4: FCS Default1
submit_jobs("pre", instance_pattern, "FCS", epsilon_values, "Pre5_fin_sparse_pre_2", binary_type, 20, config_default)


# Mixing test

using PPMP
include("scripts/run_experiments.jl")

# Configuration: No presolve, No merge, max time
config_base = ["--is-presolve-flowcut", "false", 
               "--merge-identical-scenarios", "false",
               "--max-time", "3600"]

# Test 0: No mixing (baseline)
config_no_mixing = vcat(config_base, ["--user-mixing-cuts-enabled", "false"])
submit_jobs("dev6", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], 
           "Mix_FCS_no_mixing_dev6", "bin_new2", 1, config_no_mixing)

# Test 1: Basic Star mixing
config_basic_star = vcat(config_base, ["--mixing-inequality-type", "basic_star"])
submit_jobs("dev6", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], 
           "Mix_FCS_basic_star_dev6", "bin_new2", 1, config_basic_star)

# Test 2: Complement mixing
config_complement = vcat(config_base, ["--mixing-inequality-type", "complement"])
submit_jobs("dev6", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], 
           "Mix_FCS_complement_dev6", "bin_new2", 1, config_complement)

# Test 3: Both mixing types
config_both = vcat(config_base, ["--mixing-inequality-type", "both"])
submit_jobs("dev6", ["test*"], "FCS", [0.0, 0.01, 0.05, 0.1, 0.2], 
           "Mix_FCS_both_dev6", "bin_new2", 1, config_both)

"""