# config/testset_settings.jl

"""Grid Data"""

# Instance parameters
"""
n/s  1/5   2/5   3/5   4/5
20     4     8     12    16
40     8     16    24    32
60     12    24    36    48
80     16    32    48    64
100    20    40    60    80
"""
const INSTANCE_SIZES = Dict(
    # n = 20 series
    "test_20_4" => Dict("n" => 20, "scenarios" => 4),     # n * 1/5
    "test_20_8" => Dict("n" => 20, "scenarios" => 8),     # n * 2/5
    "test_20_12" => Dict("n" => 20, "scenarios" => 12),   # n * 3/5
    "test_20_16" => Dict("n" => 20, "scenarios" => 16),   # n * 4/5

    # n = 40 series
    "test_40_8" => Dict("n" => 40, "scenarios" => 8),     # n * 1/5
    "test_40_16" => Dict("n" => 40, "scenarios" => 16),   # n * 2/5
    "test_40_24" => Dict("n" => 40, "scenarios" => 24),   # n * 3/5
    "test_40_32" => Dict("n" => 40, "scenarios" => 32),   # n * 4/5

    # n = 60 series
    "test_60_12" => Dict("n" => 60, "scenarios" => 12),   # n * 1/5
    "test_60_24" => Dict("n" => 60, "scenarios" => 24),   # n * 2/5
    "test_60_36" => Dict("n" => 60, "scenarios" => 36),   # n * 3/5
    "test_60_48" => Dict("n" => 60, "scenarios" => 48),   # n * 4/5

    # n = 80 series
    "test_80_16" => Dict("n" => 80, "scenarios" => 16),   # n * 1/5
    "test_80_32" => Dict("n" => 80, "scenarios" => 32),   # n * 2/5
    "test_80_48" => Dict("n" => 80, "scenarios" => 48),   # n * 3/5
    "test_80_64" => Dict("n" => 80, "scenarios" => 64),   # n * 4/5

    # n = 100 series
    "test_100_20" => Dict("n" => 100, "scenarios" => 20), # n * 1/5
    "test_100_40" => Dict("n" => 100, "scenarios" => 40), # n * 2/5
    "test_100_60" => Dict("n" => 100, "scenarios" => 60), # n * 3/5
    "test_100_80" => Dict("n" => 100, "scenarios" => 80)  # n * 4/5
)

const EDGE_DENSITIES = [0.3, 0.5, 0.7, 0.9]
# const EDGE_DENSITIES = [0.1,0.3, 0.5, 0.7, 0.9]
const FAILURE_PROBABILITIES = [0.1, 0.2, 0.3]
const EPSILON_VALUES = [0.05, 0.1, 0.2]
const RANDOM_SEEDS = 1:5


"""Expl Data"""

# Explorative test instances focusing on method 2's advantages
const EXPLORATIVE_INSTANCE_SIZES = Dict(
    # n = 100 series (focus on larger instances)
    "test_100_20" => Dict("n" => 100, "scenarios" => 20),  # n * 1/5
    "test_100_40" => Dict("n" => 100, "scenarios" => 40),  # n * 2/5
    
    # n = 150 series
    "test_150_30" => Dict("n" => 150, "scenarios" => 30),  # n * 1/5
    "test_150_60" => Dict("n" => 150, "scenarios" => 60),  # n * 2/5
    
    # n = 200 series
    "test_200_40" => Dict("n" => 200, "scenarios" => 40),  # n * 1/5
    "test_200_80" => Dict("n" => 200, "scenarios" => 80),  # n * 2/5
)

# Refined density values focusing on high density advantages
const EXPLORATIVE_EDGE_DENSITIES = [0.7, 0.8, 0.9, 0.95]

# More granular failure probabilities in the successful range
const EXPLORATIVE_FAILURE_PROBABILITIES = [0.05, 0.1, 0.15, 0.2]

# Epsilon values that showed promise
const EXPLORATIVE_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const EXPLORATIVE_RANDOM_SEEDS = 1:10



"""Large Data"""

# Explorative test instances focusing on method 2's advantages
const LARGE_INSTANCE_SIZES = Dict(
    # n = 100 series (focus on larger instances)
    "test_100_10" => Dict("n" => 100, "scenarios" => 10),
    "test_100_20" => Dict("n" => 100, "scenarios" => 20),
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
    "test_100_500" => Dict("n" => 100, "scenarios" => 500),
    
    
    # n = 200 series
    "test_200_10" => Dict("n" => 100, "scenarios" => 10),
    "test_200_20" => Dict("n" => 100, "scenarios" => 20),
    "test_200_50" => Dict("n" => 100, "scenarios" => 50),
    "test_200_100" => Dict("n" => 100, "scenarios" => 100),
    "test_200_200" => Dict("n" => 100, "scenarios" => 200),
    "test_200_500" => Dict("n" => 100, "scenarios" => 500),
)

# Refined density values focusing on high density advantages
const LARGE_EDGE_DENSITIES = [0.2, 0.5, 0.8]

# More granular failure probabilities in the successful range
const LARGE_FAILURE_PROBABILITIES = [0.001, 0.005, 0.01]

# Epsilon values that showed promise
const LARGE_EPSILON_VALUES = [0.05, 0.1, 0.2]

# More seeds for better statistical significance
const LARGE_RANDOM_SEEDS = 1:5


"""Developing Data"""

const DEV_INSTANCE_SIZES = Dict(
    # n = 20 series
    # "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    # "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    # "test_20_200" => Dict("n" => 20, "scenarios" => 200),
    # "test_20_500" => Dict("n" => 20, "scenarios" => 500),

    # # n = 40 series
    # "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    # "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    # "test_40_200" => Dict("n" => 40, "scenarios" => 200),
    # "test_40_500" => Dict("n" => 40, "scenarios" => 500),

    # # n = 60 series
    # "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    # "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    # "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    # "test_60_500" => Dict("n" => 60, "scenarios" => 500),
    
    # # n = 80 series
    # "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    # "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    # "test_80_200" => Dict("n" => 80, "scenarios" => 200),
    # "test_80_500" => Dict("n" => 80, "scenarios" => 500),

    # n = 100 series
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
    "test_100_500" => Dict("n" => 100, "scenarios" => 500),
)

# Refined density values focusing on high density advantages
const DEV_EDGE_DENSITIES = [0.2, 0.5, 0.8]

# More granular failure probabilities in the successful range
const DEV_FAILURE_PROBABILITIES = [0.0001, 0.005, 0.01]

# Epsilon values that showed promise
const DEV_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const DEV_RANDOM_SEEDS = 1:5


"""semi-Uniform Data"""

const SUNI_INSTANCE_SIZES = Dict(
    # n = 20 series
    "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    "test_20_200" => Dict("n" => 20, "scenarios" => 200),
    "test_20_500" => Dict("n" => 20, "scenarios" => 500),

    # n = 40 series
    "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    "test_40_200" => Dict("n" => 40, "scenarios" => 200),
    "test_40_500" => Dict("n" => 40, "scenarios" => 500),

    # n = 60 series
    "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    "test_60_500" => Dict("n" => 60, "scenarios" => 500),
    
    # n = 80 series
    "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    "test_80_200" => Dict("n" => 80, "scenarios" => 200),
    "test_80_500" => Dict("n" => 80, "scenarios" => 500),

    # n = 100 series
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
    "test_100_500" => Dict("n" => 100, "scenarios" => 500),
)

# Refined density values focusing on high density advantages
const SUNI_EDGE_DENSITIES = [0.2, 0.5, 0.8]

# More granular failure probabilities in the successful range
const SUNI_FAILURE_PROBABILITIES = [0.1]

# Epsilon values that showed promise
const SUNI_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const SUNI_RANDOM_SEEDS = 1:5

"""Developing Data 2 """

const DEV2_INSTANCE_SIZES = Dict(
    # n = 20 series
    "test_20_20" => Dict("n" => 20, "scenarios" => 20),
    "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    "test_20_200" => Dict("n" => 20, "scenarios" => 200),

    # n = 40 series
    "test_40_20" => Dict("n" => 40, "scenarios" => 20),
    "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    "test_40_200" => Dict("n" => 40, "scenarios" => 200),

    # n = 60 series
    "test_60_20" => Dict("n" => 60, "scenarios" => 20),
    "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    
    # n = 80 series
    "test_80_20" => Dict("n" => 80, "scenarios" => 20),
    "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    "test_80_200" => Dict("n" => 80, "scenarios" => 200),

    # n = 100 series
    "test_100_20" => Dict("n" => 100, "scenarios" => 20),
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
)

# Refined density values focusing on high density advantages
const DEV2_EDGE_DENSITIES = [0.3, 0.5, 0.7, 0.9]

# More granular failure probabilities in the successful range
const DEV2_FAILURE_PROBABILITIES = [0.002, 0.005, 0.01, 0.02]

# Epsilon values that showed promise
const DEV2_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const DEV2_RANDOM_SEEDS = 1:5

"""Developing Data 3 """

const DEV3_INSTANCE_SIZES = Dict(
    # n = 20 series
    "test_20_20" => Dict("n" => 20, "scenarios" => 20),
    "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    "test_20_200" => Dict("n" => 20, "scenarios" => 200),

    # n = 40 series
    "test_40_20" => Dict("n" => 40, "scenarios" => 20),
    "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    "test_40_200" => Dict("n" => 40, "scenarios" => 200),

    # n = 60 series
    "test_60_20" => Dict("n" => 60, "scenarios" => 20),
    "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    
    # n = 80 series
    "test_80_20" => Dict("n" => 80, "scenarios" => 20),
    "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    "test_80_200" => Dict("n" => 80, "scenarios" => 200),

    # n = 100 series
    "test_100_20" => Dict("n" => 100, "scenarios" => 20),
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
)

# Refined density values focusing on high density advantages
const DEV3_EDGE_DENSITIES = [0.5]

# More granular failure probabilities in the successful range
const DEV3_FAILURE_PROBABILITIES = [0.005, 0.01]

# Epsilon values that showed promise
const DEV3_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const DEV3_RANDOM_SEEDS = 1:5


"""Developing Data 4 """

const DEV4_INSTANCE_SIZES = Dict(
    # n = 20 series
    # "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    # "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    # "test_20_200" => Dict("n" => 20, "scenarios" => 200),
    # "test_20_500" => Dict("n" => 20, "scenarios" => 500),

    # # n = 40 series
    # "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    # "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    # "test_40_200" => Dict("n" => 40, "scenarios" => 200),
    # "test_40_500" => Dict("n" => 40, "scenarios" => 500),

    # # n = 60 series
    # "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    # "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    # "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    # "test_60_500" => Dict("n" => 60, "scenarios" => 500),
    
    # # n = 80 series
    # "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    # "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    # "test_80_200" => Dict("n" => 80, "scenarios" => 200),
    # "test_80_500" => Dict("n" => 80, "scenarios" => 500),

    # n = 100 series
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
    "test_100_500" => Dict("n" => 100, "scenarios" => 500),
)

# Refined density values focusing on high density advantages
const DEV4_EDGE_DENSITIES = [0.05, 0.1, 0.2]

# More granular failure probabilities in the successful range
const DEV4_FAILURE_PROBABILITIES = [0.0001, 0.005, 0.01]

# Epsilon values that showed promise
const DEV4_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const DEV4_RANDOM_SEEDS = 1:5


"""semi-Uniform Data 2"""

const SUNI2_INSTANCE_SIZES = Dict(
    # n = 20 series
    "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    "test_20_200" => Dict("n" => 20, "scenarios" => 200),
    "test_20_500" => Dict("n" => 20, "scenarios" => 500),

    # n = 40 series
    "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    "test_40_200" => Dict("n" => 40, "scenarios" => 200),
    "test_40_500" => Dict("n" => 40, "scenarios" => 500),

    # n = 60 series
    "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    "test_60_500" => Dict("n" => 60, "scenarios" => 500),
    
    # n = 80 series
    "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    "test_80_200" => Dict("n" => 80, "scenarios" => 200),
    "test_80_500" => Dict("n" => 80, "scenarios" => 500),

    # n = 100 series
    "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
    "test_100_500" => Dict("n" => 100, "scenarios" => 500),
)

# Refined density values focusing on high density advantages
const SUNI2_EDGE_DENSITIES = [0.2, 0.3, 0.4]

# More granular failure probabilities in the successful range
const SUNI2_FAILURE_PROBABILITIES = [0.1]

# Epsilon values that showed promise
const SUNI2_EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# More seeds for better statistical significance
const SUNI2_RANDOM_SEEDS = 1:5


"""Presolve Test Data"""

const PRE_INSTANCE_SIZES = Dict(
    # n = 20 series
    # "test_20_50" => Dict("n" => 20, "scenarios" => 50),
    # "test_20_100" => Dict("n" => 20, "scenarios" => 100),
    # "test_20_200" => Dict("n" => 20, "scenarios" => 200),
    # "test_20_500" => Dict("n" => 20, "scenarios" => 500),

    # # n = 40 series
    # "test_40_50" => Dict("n" => 40, "scenarios" => 50),
    # "test_40_100" => Dict("n" => 40, "scenarios" => 100),
    # "test_40_200" => Dict("n" => 40, "scenarios" => 200),
    # "test_40_500" => Dict("n" => 40, "scenarios" => 500),

    # # n = 60 series
    # "test_60_50" => Dict("n" => 60, "scenarios" => 50),
    # "test_60_100" => Dict("n" => 60, "scenarios" => 100),
    # "test_60_200" => Dict("n" => 60, "scenarios" => 200),
    # "test_60_500" => Dict("n" => 60, "scenarios" => 500),
    
    # # n = 80 series
    # "test_80_50" => Dict("n" => 80, "scenarios" => 50),
    # "test_80_100" => Dict("n" => 80, "scenarios" => 100),
    # "test_80_200" => Dict("n" => 80, "scenarios" => 200),
    # "test_80_500" => Dict("n" => 80, "scenarios" => 500),

    # # n = 100 series
    # "test_100_50" => Dict("n" => 100, "scenarios" => 50),
    # "test_100_100" => Dict("n" => 100, "scenarios" => 100),
    "test_100_200" => Dict("n" => 100, "scenarios" => 200),
    "test_100_500" => Dict("n" => 100, "scenarios" => 500),
    "test_100_1000" => Dict("n" => 100, "scenarios" => 1000),
    "test_100_2000" => Dict("n" => 100, "scenarios" => 2000),
)

# Edge density values - keep high to enable more possible dominance
const PRE_EDGE_DENSITIES = [0.5]

# Base failure probabilities (these will appear in filepath)
const PRE_BASE_FAILURE_PROBS = [0.1]

# Scenario structure configuration
const PRE_SCENARIO_CONFIG = Dict(
    "core_ratio" => 0.05,     # 5% scenarios will be core sets (dominators)
    "extended_ratio" => 0.20,  # 20% scenarios will be extended sets (dominated)
    "random_ratio" => 0.75,   # 75% scenarios will be random
    
    # Failure set sizes (as proportion of total edges)
    "core_failure_ratio" => 0.05,      # Core sets have 5% edges failed
    "extended_failure_ratio" => 0.005,   # Extended sets add 0.5% more edges failures
    "random_failure_range" => (0.001, 0.002)  # Random sets fail at a tiny ratio
)

# Random seeds for reproducibility
const PRE_RANDOM_SEEDS = 1:5


"""
Test set configuration mapping
"""
const TESTSET_CONFIGS = Dict{String, NamedTuple{(:dir, :sizes, :densities, :probs, :eps, :seeds), 
                                               Tuple{String, Dict, Vector{Float64}, Vector{Float64}, Vector{Float64}, UnitRange{Int}}}}(
    "grid" => (
        dir = "data/grid",
        sizes = INSTANCE_SIZES,
        densities = EDGE_DENSITIES,
        probs = FAILURE_PROBABILITIES,
        eps = EPSILON_VALUES,
        seeds = RANDOM_SEEDS
    ),
    "expl" => (
        dir = "data/expl",
        sizes = EXPLORATIVE_INSTANCE_SIZES,
        densities = EXPLORATIVE_EDGE_DENSITIES,
        probs = EXPLORATIVE_FAILURE_PROBABILITIES,
        eps = EXPLORATIVE_EPSILON_VALUES,
        seeds = EXPLORATIVE_RANDOM_SEEDS
    ),
    "large" => (
        dir = "data/large",
        sizes = LARGE_INSTANCE_SIZES,
        densities = LARGE_EDGE_DENSITIES,
        probs = LARGE_FAILURE_PROBABILITIES,
        eps = LARGE_EPSILON_VALUES,
        seeds = LARGE_RANDOM_SEEDS
    ),
    "dev" => (
        dir = "data/dev",
        sizes = DEV_INSTANCE_SIZES,
        densities = DEV_EDGE_DENSITIES,
        probs = DEV_FAILURE_PROBABILITIES,
        eps = DEV_EPSILON_VALUES,
        seeds = DEV_RANDOM_SEEDS
    ),
    "dev2" => (
        dir = "data/dev2",
        sizes = DEV2_INSTANCE_SIZES,
        densities = DEV2_EDGE_DENSITIES,
        probs = DEV2_FAILURE_PROBABILITIES,
        eps = DEV2_EPSILON_VALUES,
        seeds = DEV2_RANDOM_SEEDS
    ),
    "dev3" => (
        dir = "data/dev3",
        sizes = DEV3_INSTANCE_SIZES,
        densities = DEV3_EDGE_DENSITIES,
        probs = DEV3_FAILURE_PROBABILITIES,
        eps = DEV3_EPSILON_VALUES,
        seeds = DEV3_RANDOM_SEEDS
    ),
    "dev4" => (
        dir = "data/dev4",
        sizes = DEV4_INSTANCE_SIZES,
        densities = DEV4_EDGE_DENSITIES,
        probs = DEV4_FAILURE_PROBABILITIES,
        eps = DEV4_EPSILON_VALUES,
        seeds = DEV4_RANDOM_SEEDS
    ),
    "suni" => (
        dir = "data/suni",
        sizes = SUNI_INSTANCE_SIZES,
        densities = SUNI_EDGE_DENSITIES,
        probs = SUNI_FAILURE_PROBABILITIES,
        eps = SUNI_EPSILON_VALUES,
        seeds = SUNI_RANDOM_SEEDS
    ),
    "suni2" => (
        dir = "data/suni2",
        sizes = SUNI2_INSTANCE_SIZES,
        densities = SUNI2_EDGE_DENSITIES,
        probs = SUNI2_FAILURE_PROBABILITIES,
        eps = SUNI2_EPSILON_VALUES,
        seeds = SUNI2_RANDOM_SEEDS
    ),
    "pre" => (
        dir = "data/pre",
        sizes = PRE_INSTANCE_SIZES,
        densities = PRE_EDGE_DENSITIES,
        probs = PRE_BASE_FAILURE_PROBS,
        eps = EPSILON_VALUES,
        seeds = PRE_RANDOM_SEEDS
    )
)

