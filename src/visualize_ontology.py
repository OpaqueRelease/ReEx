import networkx as nx


g = nx.read_edgelist("mini_synthetic.edgelist" , create_using=nx.DiGraph)
nx.write_gexf(g, "mini_ontology.gexf")
