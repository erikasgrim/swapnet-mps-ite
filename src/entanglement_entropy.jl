using HDF5, Plots, ITensorMPS, ITensors, Statistics, LaTeXStrings, CSV, Tables

include("../src/utilities.jl")

default_entanglement_results_path(chi, graph_type, architecture, order, idx) =
    "results/portfolio/MPS_dtau10/ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.042_rho1.0_$(architecture)_chi$(chi)_$(order)/results.h5"

const DEFAULT_CHI_VALUES = (8, 16, 32, 64, 128)
const DEFAULT_GRAPH_TYPES = ("3Reg", "ER", "SK")
const DEFAULT_ARCHITECTURES = ("quadratic", "triangular")
const DEFAULT_ORDERINGS = ("shuffle",)
const DEFAULT_IDX_RANGE = 0:0

"""
    compute_and_save_entanglement_entropy(chi_values, graph_types, architectures, orderings, idx_range;
        path_builder=default_entanglement_results_path)

Compute the bipartite entanglement entropy for every MPS stored in the HDF5 result
files defined by `path_builder`. The function skips files that already contain an
`entanglement_entropy` dataset and appends the computed entropies to the HDF5 file.

# Arguments
- `chi_values`: Bond dimensions to iterate over.
- `graph_types`: Graph families present in the results.
- `architectures`: Ansatz identifiers.
- `orderings`: Ordering strategies used while generating the data.
- `idx_range`: Range of instance indices associated with a result file.

# Keyword Arguments
- `path_builder`: Function that maps `(chi, graph_type, architecture, order, idx)` to the HDF5 path.
"""
function compute_and_save_entanglement_entropy(
    chi_values,
    graph_types,
    architectures,
    orderings,
    idx_range;
    path_builder=default_entanglement_results_path,
)
    for chi in chi_values
        for graph_type in graph_types
            for architecture in architectures
                for order in orderings
                    for idx in idx_range
                        if graph_type == "SK" && order == "fiedler"
                            continue
                        end

                        path = path_builder(chi, graph_type, architecture, order, idx)

                        if !isfile(path)
                            println("Result file not found at $(path); skipping.")
                            continue
                        end

                        should_skip = h5open(path, "r") do file
                            haskey(file, "entanglement_entropy")
                        end

                        if should_skip
                            println("Entanglement entropy already exists for $(path); skipping.")
                            continue
                        end

                        z_expvals = h5read(path, "z_expvals")
                        Nsites = size(z_expvals, 1)
                        Nsteps = size(z_expvals, 2)
                        entanglement_entropy_matrix = zeros(Nsteps, Nsites - 1)

                        for i in 0:Nsteps-1
                            mps = h5open(path, "r") do file
                                read(file["psi_$i"], "MPS", MPS)
                            end
                            entropies = get_all_entanglement_entropies(mps)
                            entanglement_entropy_matrix[i + 1, 1:Nsites - 1] = entropies
                        end

                        h5open(path, "r+") do file
                            write(file, "entanglement_entropy", entanglement_entropy_matrix)
                        end

                        println("Saved entanglement entropy for $(path)")
                    end
                end
            end
        end
    end
end

compute_and_save_entanglement_entropy(
    DEFAULT_CHI_VALUES,
    DEFAULT_GRAPH_TYPES,
    DEFAULT_ARCHITECTURES,
    DEFAULT_ORDERINGS,
    DEFAULT_IDX_RANGE,
)
