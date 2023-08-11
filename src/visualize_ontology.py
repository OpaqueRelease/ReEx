import networkx as nx


g = nx.read_edgelist("synthetic.edgelist" , create_using=nx.DiGraph)
nx.write_gexf(g, "ontology.gexf")
