# Contrastive-Learning-for-Graph-Based-Biological-Interaction-Discovery

## Overview
This is the code repository for the article entitled: "Contrastive Learning for Graph-Based Biological Interaction Discovery: Insights from Oncologic Pathways"

## Discovered Pathways
<img src="./figures/mean_avg_subgraph.jpg" alt="Sample Figure" width="400" /> <img src="./figures/weighted_avg_subgraph.jpg" alt="Sample Figure" width="400" />


## Trends in Clinical Interests of Discovered Biomarkers
![Sample Figure](./figures/group_3.jpg)
![Sample Figure](./figures/group_1.jpg)
![Sample Figure](./figures/group_2.jpg)


## Code Usage
### Extract targeting interaction networks and its adjacency matrix
```python
python 0_data_preprocessing.py
```
### Contrastive learning on Isomorphic Graphs
```python
python 1_cl_graph.py
```
### Assembling synthesized networks
```python
python 2_ensemble_result.py
```
### Model Architecture
```python
model.py
```

