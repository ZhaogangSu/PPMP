# scripts/precompile/create_sysimgae.jl
using PackageCompiler

# PackageCompiler.compile_incremental(:PPMP)


create_sysimage(sysimage_path="lib/PPMP.so", precompile_execution_file="scripts/precompile/precompile.jl")