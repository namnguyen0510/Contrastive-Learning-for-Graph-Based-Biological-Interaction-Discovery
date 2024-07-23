import pandas as pd 
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np

paths = [os.path.join('_results_',x) for x in sorted(os.listdir('_results_'))]
print(paths)
n_layers = [x.split('-')[0] for x in sorted(os.listdir('_results_'))]
loss_func = [x.split('-')[1] for x in sorted(os.listdir('_results_'))]
opt_func = [x.split('-')[2] for x in sorted(os.listdir('_results_'))]


DFS = []
SCORES = []
for path in tqdm.tqdm(paths):
    DFS.append(pd.read_csv(os.path.join(path,'0_result_synthesized_graph.csv')))
    SCORES.append(pd.read_csv(os.path.join(path,'exp_hist.csv')).iloc[:,1].min())
print(DFS)
print(SCORES)
cl_df = pd.DataFrame({
    'Layers': n_layers,
    'Obj. Func.': loss_func,
    'Opt. Func.': opt_func,
    'CL Loss': SCORES
})
# Create subplots
cl_df = cl_df[cl_df['Opt. Func.'] != 'sgd']
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# Plot 1: Layers vs CL Loss
sns.barplot(ax=axes[0], x='Layers', y='CL Loss', data=cl_df, palette='Set1')
axes[0].set_xlabel('Layers', weight = 'bold')
axes[0].set_ylabel('CL Loss', weight = 'bold')

# Plot 2: Obj. Func. vs CL Loss
sns.barplot(ax=axes[1], x='Obj. Func.', y='CL Loss', data=cl_df, palette='Set2')
axes[1].set_xlabel('Objective Function', weight = 'bold')
axes[1].set_ylabel('CL Loss', weight = 'bold')

# Plot 3: Opt. Func. vs CL Loss
sns.barplot(ax=axes[2], x='Opt. Func.', y='CL Loss', data=cl_df, palette='tab10')
axes[2].set_xlabel('Optimizer Function', weight = 'bold')
axes[2].set_ylabel('CL Loss', weight = 'bold')

# Adjust layout
plt.tight_layout()
plt.savefig('result_cl_training.pdf', dpi = 600)
plt.show()


SCORES = np.array(SCORES)
SCORES = (SCORES - SCORES.min())/(SCORES.max()-SCORES.min())
SCORES = 1-SCORES

def get_ensemble(dfs, scores, method = 'average'):
    if method == 'average':
        df = dfs[0]
        for i in range(1, len(dfs)):
            df += dfs[i]
        df = df/len(dfs)
        return df
    elif method == 'weighted_average':
        df = dfs[0]*scores[0]
        for i in range(1, len(dfs)):
            df += dfs[i]*scores[i]
        df = df/len(dfs)
        return df

get_ensemble(DFS, SCORES, 'average').to_csv('mean_avg.csv', index = False)
get_ensemble(DFS, SCORES, 'weighted_average').to_csv('weighted_avg.csv', index = False)