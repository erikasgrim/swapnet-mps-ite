using Graphs

# Create 3Reg
g = Graphs.random_regular_graph(6, 3)

# print edges
for e in edges(g)
    println(e)
end

# Check if it has a edge (1, 2)
has_edge(g, 3, 1)
