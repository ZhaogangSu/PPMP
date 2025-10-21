# scripts/precompile/create_app.jl

using PackageCompiler

create_app(
    ".", # path to  PPMP package
    # "bin";  # output directory
    # "bin_force";  # output directory
    "bin_new3";  # output directory
    # "bin";  # output directory
    force=true,  # overwrite existing files
    # incremental=false,  # create from scratch for smaller size
    # filter_stdlibs=true,  # only include needed standard libraries
    precompile_execution_file="scripts/precompile/precompile.jl"
)