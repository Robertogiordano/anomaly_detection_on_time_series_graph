# Anomaly Detection on Time Series Graphs

Anomaly detection in IoT time series data is essential for identifying irregularities and potential
cyberattacks. This work focuses on leveraging spatial and temporal correlations to enhance
detection accuracy. Spatial correlation reflects the similarity of data from sensors in close
proximity, such as temperature sensors recording similar values in the same area. Temporal
correlation ensures consistency in data trends over time, avoiding unexpected spikes or irregular
behavior.

In typical scenarios, neighboring sensors produce similar data that evolves coherently, while
anomalies disrupt this pattern. To address this, I propose a method that constructs graphs based
on spatial and temporal similarities among sensors. These graphs evolve over time, capturing the
dynamics of IoT systems. Anomalies manifest as deviations in these patterns, signaling potential
attacks or irregularities.

Our objective is to develop an advanced anomaly detection system using Spatial-Temporal Graph
Autoencoders (STGAE). This approach harnesses the power of graph-based models to capture and
analyze correlations in both space and time, enabling the detection of anomalies with high
precision. This method provides a robust framework for monitoring IoT systems, ensuring their
security and reliability in real-world applications.


# Dataset Source
The original dataset is sourced from [NASA Earthdata](https://disc.gsfc.nasa.gov/datasets/SNDRSNIML3CMCCPN_2/summary?keywords=Average%20Precipitation).

# Repository Structure

This repository contains all the scripts, notebooks, and data necessary to preprocess, train, evaluate, and analyze a spatio-temporal anomaly detection model using graph-based techniques. Below is a detailed description of the structure of the repository:

```
.
├── 0_NASA_analysis.ipynb
├── 1a_preprocess.py
├── 1_export_xarray_csv.py
├── 2_analysis_and_preprocessing.ipynb
├── 3_training.ipynb
├── 4_analysis_and_evaluation.ipynb
├── corrupt_data_scripts/
├── dashboards/
├── data/
├── discarded_models/
├── earth_science_env.yml
├── folder_structure.txt
├── new_models/
├── node_connections/
├── public/
├── README.md
├── Report.pdf
├── support_csv/
└── utils/
```

---

### **Files and Folders**

#### **Main Notebooks and Scripts**
- **`0_NASA_analysis.ipynb`**: Initial exploratory analysis on the NASA dataset.
- **`1a_preprocess.py`**: Python script for preprocessing raw data files.
- **`1_export_xarray_csv.py`**: Converts processed data into CSV format for further analysis.
- **`2_analysis_and_preprocessing.ipynb`**: Detailed analysis and preprocessing of the dataset, including feature engineering.
- **`3_training.ipynb`**: Training notebook for the anomaly detection model.
- **`4_analysis_and_evaluation.ipynb`**: Post-training analysis and evaluation of model performance.

#### **Supporting Files and Scripts**
- **`corrupt_data_scripts/`**: Scripts to introduce artificial corruption into the dataset:
  - `corrupt_data.py`: Temporal corruption script.
  - `corrupt_data_space.py`: Temporal corruption script on filtered spatial nodes.
- **`dashboards/`**: Scripts to generate interactive dashboards for data and model analysis:
  - `dashboard_analysis.py`: Model results analysis-specific dashboard.
  - `dashboard.py`: General dashboard for data visualization.

#### **Data**
- **`data/`**: Contains raw, preprocessed, and augmented datasets:
  - `data.csv`: Main dataset generated after *1_export_xarray_csv.py*.
  - `data_preprocessed_2013_2015.csv`, `data_preprocessed.csv`: Preprocessed datasets genereated in *2_analysis_and_preprocessing.ipynb*.
  - `subdata/`, `subdata_augmented/`, `subdata_corrupted/`: Filtered dataset on *data_preprocessed.csv* having only data on 2015. 
  - `subdata_extended/`, `subdata_extended_augmented/`, `subdata_extended_corrupted/`: Filtered dataset on *data_preprocessed_2013_2015.csv*. 
  - `correlation_matrix.csv`: Precomputed correlation matrix for features.
  - `pairwise_distances.csv`: Pairwise distances between nodes for graph creation.
  - `unique_coords.csv`: Unique selected geographical coordinates in the dataset.

#### **Models**
- **`discarded_models/`**: Contains experimental model architectures that were tested and discarded.
- **`new_models/`**: Includes current working models and architectures for anomaly detection.

#### **Graph Data**
- **`node_connections/`**: Preprocessed graph-related data:
  - `edge_index.pt`, `edge_weight.pt`, `node_features.pt`: Files containing graph edge indices, weights, and node features.
  - `node_ids.pkl`, `node_mapping.pkl`: Node ID mappings.

#### **Utilities**
- **`utils/`**: Utility scripts and notebooks for data analysis and processing:
  - `create_pairwise_distances.py`: Script to compute pairwise distances for nodes.

#### **Environment and Documentation**
- **`earth_science_env.yml`**: Conda environment file with all dependencies for reproducing the analysis.
- **`Report.pdf`**: Detailed report summarizing the methodology, experiments, and results.
- **`README.md`**: This file, providing an overview of the repository structure.

#### **Support Data**
- **`support_csv/`**: Contains smaller CSV files for specific analysis purposes.

---

### **How to Use**

1. **Setup Environment**: Install dependencies using the `earth_science_env.yml` file:
   ```
   conda env create -f earth_science_env.yml
   conda activate earth_science_env
   ```

2. **Preprocessing**:
   - Use `1a_preprocess.py` and `2_analysis_and_preprocessing.ipynb` to preprocess the raw dataset.

3. **Training**:
   - Run `3_training.ipynb` to train the spatio-temporal anomaly detection model.

4. **Evaluation**:
   - Analyze results using `4_analysis_and_evaluation.ipynb`.

5. **Visualizations and Dashboards**:
   - Explore visualizations in the model'sfolder or run the dashboards in `dashboards/`.

For more details, refer to the `Report.pdf`.

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
