"""
    load_ising(path)

Load an Ising instance encoded as JSON with coupling matrix `J`, optional single-site fields
`h`, and constant offset `c`. Keys are assumed to be zero-based; the function shifts them to
Julia's 1-based indexing.

# Arguments
- `path`: Filesystem path to the JSON input.

# Returns
A tuple `(J, h, offset)` where `J` is a dictionary keyed by `(i, j)`, `h` is a dictionary keyed
by site index, and `offset` is the scalar constant term.
"""
function load_ising(path::String)
	data = JSON.parsefile(path)

	Jdict = data["J"]
	#hdict = data["h"]
	hdict = haskey(data, "h") ? data["h"] : Dict{String, Float64}()
	offset = get(data, "c", 0.0)  # Optional constant term

	J = Dict{Tuple{Int, Int}, Float64}()
	for (k, v) in Jdict
		i, j = split(k, ",")             # ["0","1"]
		i = parse(Int, i) + 1            # shift to 1-based
		j = parse(Int, j) + 1
		J[(i, j)] = v
	end

	h = Dict{Int, Float64}()
	for (k, v) in hdict
		i = parse(Int, k) + 1            # shift to 1-based
		h[i] = v
	end

	return J, h, offset
end

"""
    state_energy(zvals, zzcorr, J, h, offset)

Compute the energy expectation value of a state using its local `Z` expectation values and
pairwise correlations.

# Arguments
- `zvals`: Vector of ⟨Z⟩ expectation values ordered by logical qubit index.
- `zzcorr`: Matrix of ⟨ZᵢZⱼ⟩ correlations matching `zvals`.
- `J`: Dictionary of pairwise couplings.
- `h`: Dictionary of on-site fields.
- `offset`: Constant term from the Ising Hamiltonian.

# Returns
Total energy as a `Float64`.
"""
function state_energy(
	zvals::Vector{Float64},
	zzcorr::Matrix{Float64},
	J::Dict{Tuple{Int, Int}, Float64},
	h::Dict{Int, Float64},
	offset::Float64,
)::Float64
	energy = offset
	N = length(zvals)
	for i in 1:N
		si = zvals[i]
		energy += get(h, i, 0.0) * si
		for j in i+1:N
			sj = zvals[j]
			energy += get(J, (i, j), 0.0) * zzcorr[i, j]
		end
	end
	return energy
end

"""
    sample_energy(sample, J, h, offset)

Evaluate the Ising energy for a concrete bitstring interpreted as spins `{0 ↦ +1, 1 ↦ -1}`.

# Arguments
- `sample`: Vector of bits (0/1) representing a spin configuration.
- `J`: Dictionary of pairwise couplings.
- `h`: Dictionary of on-site fields.
- `offset`: Constant term from the Hamiltonian.

# Returns
Energy of the sample as a `Float64`.
"""
function sample_energy(
	sample::Vector{Int},
	J::Dict{Tuple{Int, Int}, Float64},
	h::Dict{Int, Float64},
	offset::Float64,
)::Float64
	energy = offset
	N = length(sample)
	for i in 1:N
		si = sample[i] == 0 ? 1 : -1
		energy += get(h, i, 0.0) * si
		for j in i+1:N
			sj = sample[j] == 0 ? 1 : -1
			energy += get(J, (i, j), 0.0) * si * sj
		end
	end
	return energy
end

"""
    is_independent_set(bitstring, edges)

Check whether the bitstring encodes an independent set on the provided edge list.

# Arguments
- `bitstring`: Vector of 0/1 values selecting vertices.
- `edges`: Vector of graph edges `(i, j)`.

# Returns
`true` when the selected vertices form an independent set.
"""
function is_independent_set(
	bitstring::Vector{Int},
	edges::Vector{Tuple{Int, Int}},
)::Bool
	for (i, j) in edges
		if bitstring[i] == 1 && bitstring[j] == 1
			return false
		end
	end
	return true
end

"""
    get_expvals(psi[, logical_qubit_order])

Collect single-site and pairwise Z expectation values for the supplied MPS. Results are
reordered according to `logical_qubit_order`.

# Arguments
- `psi`: Matrix product state.
- `logical_qubit_order`: Optional mapping from physical positions to logical indices.

# Returns
Tuple `(z_expvals, zz_expvals)` with reordered expectation values.
"""
function get_expvals(psi, logical_qubit_order = collect(1:length(psi)))
	z_expvals = 2 * expect(psi, "Sz")
	zz_expvals = 4 * correlation_matrix(psi, "Sz", "Sz")
	# Reorder zvals and zzcorr
	logical_to_site = invperm(logical_qubit_order)
	z_expvals = z_expvals[logical_to_site]
	zz_expvals = zz_expvals[logical_to_site, logical_to_site]
	return z_expvals, zz_expvals
end

"""
    get_samples(psi, N_s, logical_qubit_order)

Draw computational basis samples from the MPS, reorder them to logical qubit order, and return
the results as a matrix where each column is a sample.

# Arguments
- `psi`: Matrix product state to sample from.
- `N_s`: Number of samples to draw.
- `logical_qubit_order`: Mapping from physical positions to logical indices.

# Returns
Matrix of size `(N, N_s)` with bit values in `{0, 1}`.
"""
function get_samples(psi, N_s, logical_qubit_order)
	psi = orthogonalize(psi, 1)
	samples = [sample(psi) .- 1 for _ in 1:N_s]
	# If logical order is permuted, reorder bits in samples
	logical_to_site = invperm(logical_qubit_order)
	for s in samples
		s[:] = s[logical_to_site]
	end
	return hcat(samples...)
end

"""
    get_entanglement_entropy(psi, b)

Compute the von Neumann entanglement entropy across bond `b` of the MPS.

# Arguments
- `psi`: Matrix product state.
- `b`: Bond index (between site `b` and `b+1`).

# Returns
Entropy value `S(ρ_b)` as a `Float64`.
"""
function get_entanglement_entropy(psi, b)
	psi = orthogonalize(psi, b)
	U, S, V = svd(psi[b], (linkinds(psi, b - 1)..., siteinds(psi, b)...))
	SvN = 0.0
	for n ∈ 1:dim(S, 1)
		p = S[n, n]^2
		SvN -= p * log(p)
	end
	return SvN
end

"""
    get_all_entanglement_entropies(psi)

Evaluate the entanglement entropy across every adjacent bipartition of the MPS.

# Arguments
- `psi`: Matrix product state.

# Returns
Vector of entropies for bonds `1:(length(psi)-1)`.
"""
function get_all_entanglement_entropies(psi)
	N = length(psi)
	entropies = Float64[]
	for b in 1:N-1
		SvN = get_entanglement_entropy(psi, b)
		push!(entropies, SvN)
	end
	return entropies
end

"""
    save_results(state_list, state_energies_list, z_expvals_list, zz_expvals_list,
                 samples_list, energy_samples_list, time_list, runtime_list, params,
                 save_dir[, file_prefix])

Persist TEBD diagnostics to `save_dir`, writing tensor data to an HDF5 archive and parameters
to `params.json`.

# Arguments
- `state_list`, `state_energies_list`, …: Recorded observables and states per TEBD step.
- `params`: Dictionary of run metadata.
- `save_dir`: Directory where files will be stored.
- `file_prefix`: Base name for the HDF5 file (defaults to `"results"`).

Returns `nothing`.
"""
function save_results(
	state_list::Vector{MPS},
	state_energies_list::Vector{Float64},
	z_expvals_list::Vector{Vector{Float64}},
	zz_expvals_list::Vector{Matrix{Float64}},
	samples_list::Vector{Matrix{Int}},
	energy_samples_list::Vector{Vector{Float64}},
	time_list::Vector{Float64},
	runtime_list::Vector{Float64},
	params::Dict{String, Any},
	save_dir::String,
	file_prefix::String = "results"
)
	z_expvals_array = reduce(hcat, z_expvals_list)
	zz_expvals_array = cat(zz_expvals_list...; dims = 3)
	energy_samples_array = reduce(hcat, energy_samples_list)
	samples_array = cat(samples_list...; dims = 3)
	# Save results to HDF5 file
	h5file = joinpath(save_dir, "$(file_prefix).h5")
	println("Saving results to $h5file")
	h5open(h5file, "w") do file
		write(file, "state_energies", state_energies_list)
		write(file, "z_expvals", z_expvals_array)
		write(file, "zz_expvals", zz_expvals_array)
		write(file, "samples", samples_array)
		write(file, "energy_samples", energy_samples_array)
		write(file, "time", time_list)
		write(file, "runtime", runtime_list)

		for (s, psi) in enumerate(state_list)
			g = create_group(file, "psi_$(s-1)")
			write(g, "MPS", psi)
		end
	end

	# Save parameters to JSON file
	paramfile = joinpath(save_dir, "params.json")
	open(paramfile, "w") do file
		JSON.print(file, params)

	end
	return nothing
end

"""
    fiedler_ordering(J)

Order qubits according to the Fiedler vector of the weighted Laplacian induced by couplings `J`.
Useful for linearising sparse interaction graphs.

# Arguments
- `J`: Dictionary whose keys `(i, j)` carry interaction strengths.

# Returns
Permutation vector listing qubit indices in the Fiedler order.
"""
function fiedler_ordering(J::Dict{Tuple{Int,Int}, Float64})
    # Determine number of qubits
    nodes = unique(vcat([i for (i,j) in keys(J)], [j for (i,j) in keys(J)]))
    N = maximum(nodes)

    # Build adjacency matrix using absolute weights
    rows, cols, vals = Int[], Int[], Float64[]
    for ((i,j), Jij) in J
        w = abs(Jij)  # use absolute value
        push!(rows, i); push!(cols, j); push!(vals, w)
        push!(rows, j); push!(cols, i); push!(vals, w)  # symmetric
    end

    A = sparse(rows, cols, vals, N, N)
    D = spdiagm(0 => vec(sum(A, dims=2)))
    L = D - A

    # Compute Fiedler vector (second smallest eigenvector)
    λ, V = Arpack.eigs(L; nev=2, which=:SM)  # smallest magnitude eigenvalues
    V = real(V)
    fiedler_vector = V[:,2]
	
    # Return permutation of nodes sorted by Fiedler vector
    ordering = sortperm(fiedler_vector)  # 1-based indices
    return ordering
end
