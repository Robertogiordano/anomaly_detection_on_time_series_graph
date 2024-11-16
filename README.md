# Anomaly Detection on Time Series Graphs

Anomaly detection in time series data from IoT systems, leveraging both spatial and temporal correlations. Specifically:

- **Spatial Correlation**: Refers to the similarity of data collected by sensors located near each other in space (e.g., temperature sensors on the same side of a room recording similar values).
- **Temporal Correlation**: Refers to the consistency of data recorded by the same device over time (e.g., absence of random spikes in recorded values).

In a non-anomalous scenario, neighboring sensors collect similar data that evolves coherently over time.

## Approach

Given a set of IoT sensors distributed in space, we can construct a graph based on the spatial and temporal similarity of the data collected by each sensor. Anomalies, such as cyberattacks, might manifest as unexpected changes in recorded metrics (e.g., sensor values). By monitoring the temporal evolution of these graphs, we can detect deviations that indicate potential attacks or irregularities in the system.

### Objective

We aim to develop an anomaly detector that exploits the spatial and temporal correlation of data using **Spatial-Temporal Graph Autoencoders (STGAE)**.

# Repository Organization

## Dataset Source
The original dataset is sourced from [NASA Earthdata](https://disc.gsfc.nasa.gov/datasets/SNDRSNIML3CMCCPN_2/summary?keywords=Average%20Precipitation).

## Data Structure
- **`/data/SNDRSNML2RMS_1-20241110_220957`**: Contains all granules from 2015, sampled daily every 3 months.
  
### Data Processing Overview
1. **Initial Analysis**:
   - Conducted in the notebook `NASA_analysis.ipynb`.
   - Outputs: `data/data.csv`, the result of an initial processing phase.

2. **Preprocessing**:
   - Performed in `anomaly_detection_on_time_series_graph.ipynb`.
   - Outputs: `data/data_preprocessed.csv`.
   - This step retains all relevant information and formats/approximates data as needed.

3. **Node Connections**:
   - The `node_connections/edge_index` file defines relationships between nodes.
   - Contains 1,657 nodes common across granules.

### Data Files
- **`data/subdata`**:
  - Contains the node features for each node for every recorded day.
  
- **`data/augmented_data`**:
  - Includes node features for all days of the year (filled for missing days).
  - Augmented through interpolation with added noise to maintain structural integrity.

## Variable Analysis
- Use `dashboard.py` for detailed variable analysis and visualization.

## Fonti
### STGNN/STGAE/GNN paper:
1. Anomaly Detection in Multi-Agent Trajectories for Automated Driving: https://proceedings.mlr.press/v164/wiederer22a/wiederer22a.pdf
2. A novel spatial–temporal generative autoencoder for wind speed uncertainty forecastin: https://www.sciencedirect.com/science/article/pii/S036054422302340X?casa_token=j9cGCl-4aLkAAAAA:hXAS6zQkpx3Cqcfmu2NzQRwO6qkoMei94EEZGlAUEbGiEtOlJzH7pw3EA60dFakVJPS8Id2rCII
3. STGNN-TTE: Travel time estimation via spatial–temporal graph neural network: https://www.sciencedirect.com/science/article/pii/S0167739X21002740?casa_token=2MazexWTIPoAAAAA:vniQuciAsY0V5NWu3js2YrBGHJI4Vro2pK8XzEZMe1UiPUZuHBiaIaMc2WY2hbill2rZbHjV4uw
4. Correlation-Aware Spatial–Temporal Graph Learning for Multivariate Time-Series Anomaly Detection: https://ieeexplore.ieee.org/abstract/document/10316684?casa_token=nVLWdmDNepYAAAAA:XIsH5Aw4Q71L8iTQ7vYuhgIgzJkDrwivHw8hi5Uyb8aRVaEkmY5YuQGOT_8tApxI552kFNr74A
5. Graph Neural Networks for Anomaly Detection in Industrial Internet of Things: https://ieeexplore.ieee.org/document/9471816
6. Anomaly detection using spatial and temporal information in multivariate time series: https://www.nature.com/articles/s41598-023-31193-8

7. Learning Convolutional Neural Networks for Graphs: https://arxiv.org/pdf/1605.05273
8. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Frameworkfor Traffic Forecasting: https://arxiv.org/pdf/1709.04875
9. STGAE: Spatial-Temporal Graph Auto-Encoder for Hand Motion Denoising: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9583812
10. Spatial-Temporal Graph Auto-Encoder for Traffic Flow Prediction: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4250436

### Tutorial vari e link utili:
1. Come leggere i file netcdf (nc): https://stackoverflow.com/questions/36360469/read-nc-netcdf-files-using-python
2. PyGod, come creare al volo una GNN/GAE: https://docs.pygod.org/en/latest/
3. Come creare una GAE/GNN con pytorch geometric: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py

### Implementazioni esistenti di STGNN:
1. https://github.com/LMissher/STGNN
2. **Spatio-Temporal Denoising Graph Autoencoder (STGAE)**: https://github.com/Yangxin666/STGAE
3. https://github.com/ZhouKanglei/STGAE
4. https://github.com/LMissher/STGNN
5. https://github.com/xiaoxiaotibaiwan/STGNN
