using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON
  
function expZZ(tau, J)
  ZZ = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
  return exp(-tau * J * ZZ)
end

function expZ(tau, h)
  Z = [1 0; 0 -1]
  return exp(-tau * h * Z)
end

function load_ising(path::String)
    data = JSON.parsefile(path)

    Jdict = data["J"]
    hdict = data["h"]

    J = Dict{Tuple{Int,Int}, Float64}()
    for (k, v) in Jdict
        i, j = split(k, ",")             # ["0","1"]
        i = parse(Int, i) + 1            # shift to 1-based
        j = parse(Int, j) + 1
        J[(i,j)] = v
    end

    h = Dict{Int, Float64}()
    for (k, v) in hdict
        i = parse(Int, k) + 1            # shift to 1-based
        h[i] = v
    end

    return J, h
end

function ising_energy(sample::Vector{Int}, J::Dict{Tuple{Int,Int}, Float64}, h::Dict{Int, Float64})::Float64
  energy = 0.0
  N = length(sample)
  for i in 1:N
    si = sample[i] == 0 ? 1 : -1
    energy += get(h, i, 0.0) * si
    for j in i+1:N
      sj = sample[j] == 0 ? 1 : -1
      energy += get(J, (i,j), 0.0) * si * sj
    end
  end
  return energy
end

function SWAP_network_block(s, J::Dict{Tuple{Int,Int}, Float64}, h::Dict{Int, Float64}, tau::Float64)
  SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
  Z = [1 0; 0 -1]
  N = length(s)

  logical_qubit_order = collect(Int, 1:N)
  gates = ITensor[]

  for _ in 1:2 # Two full sweeps to get logical qubits back to original order

    # Apply two-qubit coupling via SWAP network
    for layer in 1:N
      if layer % 2 == 1
        pairs = [(i, i+1) for i in 1:2:N-1 if i+1 <= N]
      else layer % 2 == 0
        pairs = [(i, i+1) for i in 2:2:N-1 if i+1 <= N]
      end

      for (a, b) in pairs
        qa = logical_qubit_order[a]
        qb = logical_qubit_order[b]
        
        coupling_weight = get(J, (min(qa,qb), max(qa,qb)), 0.0)
        if coupling_weight != 0.0
          gate = op(expZZ(tau, coupling_weight), s[a], s[b])
          push!(gates, gate)
        end

        SWAP_gate = ITensorMPS.op(SWAP, s[a], s[b])
        push!(gates, SWAP_gate)

        logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
      end
    end

    # Apply single-qubit fields
    for i in 1:N
      qi = logical_qubit_order[i]
      field_strength = get(h, qi, 0.0)
      if field_strength != 0.0
        gate = op(expZ(tau, field_strength), s[i])
        push!(gates, gate)
      end
    end

  end

  return gates
end

function Hadamard_block(s)
  H = (1/sqrt(2)) * [1 1; 1 -1]
  N = length(s)
  gates = ITensor[]
  for i in 1:N
    gate = op(H, s[i])
    push!(gates, gate)
  end
  return gates
end

let
  SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
  H = (1/sqrt(2)) * [1 1; 1 -1]
  Z = [1 0; 0 -1]
  
  # TEBD parameters
  cutoff = 1E-4
  tau = 3.0
  ttotal = 3.0
  chi = 32

  # Load JSON file 
  path = "data/Ising/ising_Ns10_Nt9_Nq1_K15_gamma0.5_zeta0.1_rho5.0.json"
  J, h = load_ising(path)
  N = maximum(collect(keys(h)))  # Number of qubits

  println("Optimizing Ising model with $N qubits and $(length(collect(keys(J)))) couplings")

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=false)
  println("Number of sites: $(length(s))")

  Hadamard_gates = Hadamard_block(s)
  TEBD_gates = SWAP_network_block(s, J, h, tau)

  # Optimization loop
  psi = productMPS(s, "Up")
  psi = apply(Hadamard_gates, psi; cutoff=cutoff, maxdim=chi)

  for t in 0.0:tau:ttotal
    println("Time: $t")
    psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)
    normalize!(psi)
  end

  # Generate N_s samples
  N_s = 1000
  psi = orthogonalize(psi, 1)
  samples = [sample(psi).-1 for _ in 1:N_s]
  energy_samples = [ising_energy(s, J, h) for s in samples]

  avg_energy = Statistics.mean(energy_samples)
  std_energy = Statistics.std(energy_samples)
  println("Average energy: $avg_energy ± $std_energy")

  # Generate random bitstrings and compute their energies
  random_samples = [rand(0:1, N) for _ in 1:N_s]
  random_energies = [ising_energy(s, J, h) for s in random_samples]

  avg_random_energy = Statistics.mean(random_energies)
  std_random_energy = Statistics.std(random_energies)
  println("Random bitstrings average energy: $avg_random_energy ± $std_random_energy")
end