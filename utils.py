import pandas as pd 
import numpy
import torch
import numpy as np


def get_symmetric(matrix):
    # Make the matrix symmetric
    symmetric_matrix = (matrix + matrix.t()) / 2.0
    return symmetric_matrix

def permute_df(df):
    n_genes = len(df) 
    gene_index = df.columns
    adj_array = df.to_numpy()
    # Create a DataFrame from the adjacency matrix
    df_adj = pd.DataFrame(adj_array, index=gene_index, columns=gene_index)
    # Permute the index
    permuted_index = np.random.permutation(gene_index)
    # Rearrange the DataFrame according to the permuted index
    df_adj_permuted = df_adj.loc[permuted_index, permuted_index]
    # Convert the permuted DataFrame back to a tensor
    adj_matrix_permuted = torch.tensor(df_adj_permuted.values, dtype=torch.float64, requires_grad=True)
    # Print the permuted index and adjacency matrix
    return permuted_index, adj_matrix_permuted