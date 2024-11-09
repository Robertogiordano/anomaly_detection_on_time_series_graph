# anomaly_detection_on_time_series_graph

Anomaly Detection su Time Series di grafi in ambito IoT, sfruttando la correlazione spaziale e temporale dei dati.  
In particolare:  
- La correlazione spaziale si riferisce alla similarità dei dati raccolti da nodi vicini tra loro nello spazio (es. sensori di temperatura nello stesso lato della stanza che misurano temperature simili).  
- La correlazione temporale, invece, riguarda la coerenza dei dati raccolti dallo stesso dispositivo nel tempo (es. assenza di picchi casuali nei valori registrati).  

In un contesto non anomalo, sensori vicini raccolgono dati simili che evolvono in modo coerente nel tempo.

Dato un determinato numero di sensori IoT distribuiti nello spazio, è possibile costruire un grafo basato sulla similarità spaziale e temporale dei dati raccolti da ciascun sensore. Le anomalie, come i cyberattacchi, possono manifestarsi come cambiamenti inaspettati nelle metriche registrate (ad esempio, nei valori dei sensori). Monitorando l'evoluzione temporale di questi grafi, si possono individuare variazioni che potrebbero indicare attacchi o altre irregolarità nel sistema.

Vorremmo sviluppare un anomaly detector che sfrutti questa correlazione spaziale e temporale dei dati utilizzando una STGNN o una STGAE. Esistono implementazioni di STGNN / STGAE in letteratura, ma andrebbero adattate al nostro specifico contesto. Se il progetto ti interessa, possiamo sentirci per una call di approfondimento.

## NASA Dataset
- https://disc.gsfc.nasa.gov/datasets/SNDRSNIML3CMCCPN_2/summary?keywords=Average%20Precipitation

## Fonti
### STGNN/STGAE/GNN paper:
1. Anomaly Detection in Multi-Agent Trajectories for Automated Driving: https://proceedings.mlr.press/v164/wiederer22a/wiederer22a.pdf
2. A novel spatial–temporal generative autoencoder for wind speed uncertainty forecastin: https://www.sciencedirect.com/science/article/pii/S036054422302340X?casa_token=j9cGCl-4aLkAAAAA:hXAS6zQkpx3Cqcfmu2NzQRwO6qkoMei94EEZGlAUEbGiEtOlJzH7pw3EA60dFakVJPS8Id2rCII
3. STGNN-TTE: Travel time estimation via spatial–temporal graph neural network: https://www.sciencedirect.com/science/article/pii/S0167739X21002740?casa_token=2MazexWTIPoAAAAA:vniQuciAsY0V5NWu3js2YrBGHJI4Vro2pK8XzEZMe1UiPUZuHBiaIaMc2WY2hbill2rZbHjV4uw
4. Correlation-Aware Spatial–Temporal Graph Learning for Multivariate Time-Series Anomaly Detection: https://ieeexplore.ieee.org/abstract/document/10316684?casa_token=nVLWdmDNepYAAAAA:XIsH5Aw4Q71L8iTQ7vYuhgIgzJkDrwivHw8hi5Uyb8aRVaEkmY5YuQGOT_8tApxI552kFNr74A
5. Graph Neural Networks for Anomaly Detection in Industrial Internet of Things: https://ieeexplore.ieee.org/document/9471816

### Tutorial vari e link utili:
1. Come leggere i file netcdf (nc): https://stackoverflow.com/questions/36360469/read-nc-netcdf-files-using-python
2. PyGod, come creare al volo una GNN/GAE: https://docs.pygod.org/en/latest/
3. Come creare una GAE/GNN con pytorch geometric: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py

### Implementazioni esistenti di STGNN:
1. https://github.com/LMissher/STGNN
2. https://github.com/Yangxin666/STGAE
3. https://github.com/ZhouKanglei/STGAE
4. https://github.com/LMissher/STGNN
5. https://github.com/xiaoxiaotibaiwan/STGNN
