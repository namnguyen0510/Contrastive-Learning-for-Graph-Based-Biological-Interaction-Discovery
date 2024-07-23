import os
import pandas as pd 
import numpy
import networkx as nx
import tqdm
import pickle

# MERGE DATASET
dfs = [pd.read_csv(os.path.join('dataset',x), sep = '\t') for x in os.listdir('dataset')]
df = pd.concat(dfs).drop_duplicates()
print(df)
df.to_csv('0_string_graph_unified.csv', index = False)

# Create a graph
G = nx.Graph()
for i in tqdm.tqdm(range(len(df))):
    node1 = df['#node1'].to_numpy()[i]
    node2 = df['node2'].to_numpy()[i]
    weight = df['combined_score'].to_numpy()[i]
    G.add_edge(node1, node2, weight=weight)

# Save the graph as a pickle file
with open('1_original_graph_{}.pkl'.format(len(df)), 'wb') as f:
    pickle.dump(G, f)
print("Graph saved!")

# Get the adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()
# Create a DataFrame from the adjacency matrix
df_adj = pd.DataFrame(adj_matrix, index=G.nodes(), columns=G.nodes())
# Save the DataFrame to a CSV file
df_adj.to_csv('2_adjacency_matrix.csv', index=False)


'''# Draw the graph
pos = nx.spring_layout(G)  # positions for all nodes
# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# Show plot
plt.show()'''