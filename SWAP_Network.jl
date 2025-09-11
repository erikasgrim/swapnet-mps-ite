using ITensors, ITensorMPS, Graphs, Plots

# Get matrix for exp(-tau w ZZ) where w is some parameter
function expZZ(tau, w)
  ZZ = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
  return exp(-tau * w * ZZ)
end

SWAP_matrix = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
H_matrix = (1/sqrt(2)) * [1 1; 1 -1]

let
  N = 200
  cutoff = 1E-4
  tau = 0.01
  ttotal = 1.0
  chi = 32

  # Generate a complete graph with 100 vertices
  g = Graphs.complete_graph(N)
  println("Generated graph with $(nv(g)) vertices and $(ne(g)) edges.")

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=false)

  Hadamard_gates = ITensor[]
  for i in 1:N
    H_gate = op(H_matrix, s[i])
    push!(Hadamard_gates, H_gate)
  end

  # Order of logical qubits starts as 1, 2, ..., N
  logical_qubit_order = collect(Int, 1:N)

  TEBD_gates = ITensor[]
  for layer in 1:N
    if layer % 2 == 1
      pairs = [(i, i+1) for i in 1:2:N-1 if i+1 <= N]
    else layer % 2 == 0
      pairs = [(i, i+1) for i in 2:2:N-1 if i+1 <= N]
    end

    for (a, b) in pairs
      qa = logical_qubit_order[a]
      qb = logical_qubit_order[b]

      if has_edge(g, qa, qb)
        TEBD_gate = op(expZZ(tau, 1.0), s[a], s[b])
        push!(TEBD_gates, TEBD_gate)
      end

      SWAP_gate = ITensorMPS.op(SWAP_matrix, s[a], s[b])
      push!(TEBD_gates, SWAP_gate)

      logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
    end
  end

  # TODO: Must apply the reverse SWAPS to return to original order

  # Final logical
  #println("Final qubit order: ", logical_qubit_order)
  psi = productMPS(s, "Up")
  psi = apply(Hadamard_gates, psi; cutoff=cutoff, maxdim=chi)

  for t in 0.0:tau:ttotal
    println("Time: $t")
    psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)
  end
end