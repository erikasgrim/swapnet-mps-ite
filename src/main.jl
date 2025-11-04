using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random, HDF5, SparseArrays, Arpack

include("utilities.jl")
include("tebd.jl")

"""
    run_parameter_sweep(; kwargs...)

Execute TEBD sweeps across graph families, bond dimensions, and network architectures, saving
each run under the configured results directory. Keyword arguments mirror the default study
parameters used in the accompanying paper.
"""
function run_parameter_sweep(;
	chi_values = [8, 16, 32, 64, 128],
	cutoff = 1e-9,  # Truncation cutoff for bond dimension
	Nsamples = 1000,
	Nsteps = 30,
	n_threads = 1,
	qubit_orderings = ["shuffle", "fiedler"],
	network_architectures = ["triangular", "quadratic"],
	graph_types = ["ER", "SK", "3Reg"],
	instance_indices = 0:9,
	directory_name = "example_run",
)
	instance_config(graph::AbstractString, idx) = begin
		data_root = joinpath(@__DIR__, "..", "data")
		results_root = joinpath(@__DIR__, "..", "results")
		if graph == "ER"
			tau = 3 / 50
			data_path = joinpath(data_root, "MaxCut", "ER", "100v", "Ising", "ising_graph$(idx).json")
			run_root = joinpath(results_root, "MaxCut", "ER", "100v", directory_name)
			name_prefix = "ising_graph$(idx)"
		elseif graph == "3Reg"
			tau = 1.0
			data_path = joinpath(data_root, "MaxCut", "3Reg", "100v", "Ising", "ising_graph$(idx).json")
			run_root = joinpath(results_root, "MaxCut", "3Reg", "100v", directory_name)
			name_prefix = "ising_graph$(idx)"
		elseif graph == "SK"
			tau = 3 / 100
			data_path = joinpath(data_root, "MaxCut", "SK", "100v", "Ising", "ising_graph$(idx).json")
			run_root = joinpath(results_root, "MaxCut", "SK", "100v", directory_name)
			name_prefix = "ising_graph$(idx)"
		elseif graph == "portfolio"
			tau = 10.0
			data_path = joinpath(data_root, "portfolio", "Ising", "ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.004_rho1.0.json")
			run_root = joinpath(results_root, "portfolio", directory_name)
			name_prefix = "ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.004_rho1.0"
		else
			error("Unknown graph type: $graph")
		end
		return tau, data_path, run_root, name_prefix
	end

	for graph in graph_types
		for chi in chi_values
			for qubit_ordering in qubit_orderings
				for network_architecture in network_architectures
					if graph == "SK" && qubit_ordering == "fiedler"
						# SK graphs are fully connected: spectral ordering not applicable
						continue
					end
					for idx in instance_indices
						tau, data_path, run_root, name_prefix = instance_config(graph, idx)
						save_dir = joinpath(run_root, "$(name_prefix)_$(network_architecture)_chi$(chi)_$(qubit_ordering)")
						run_TEBD(
							data_path;
							chi = chi,
							cutoff = cutoff,
							tau = tau,
							Nsamples = Nsamples,
							Nsteps = Nsteps,
							n_threads = n_threads,
							qubit_ordering = qubit_ordering,
							network_architecture = network_architecture,
							save_dir = save_dir,
							seed = idx,
						)
					end
				end
			end
		end
	end
	return nothing
end

let
	# Example usage parameter sweep:
	# run_parameter_sweep()

	# Single example run:
	# run_TEBD(
	# 	joinpath(@__DIR__, "..", "data", "MaxCut", "ER", "100v", "Ising", "ising_graph0.json");
	# 	save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "ER", "100v", "example_run"),
	# 	chi = 16,
	# 	tau = 3 / 50,
	# )
end
