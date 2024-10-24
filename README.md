# anomaly_detection_on_time_series_graph

Anomaly Detection su Time Series di grafi in ambito IoT, sfruttando la correlazione spaziale e temporale dei dati.  
In particolare:  
- La correlazione spaziale si riferisce alla similarità dei dati raccolti da nodi vicini tra loro nello spazio (es. sensori di temperatura nello stesso lato della stanza che misurano temperature simili).  
- La correlazione temporale, invece, riguarda la coerenza dei dati raccolti dallo stesso dispositivo nel tempo (es. assenza di picchi casuali nei valori registrati).  

In un contesto non anomalo, sensori vicini raccolgono dati simili che evolvono in modo coerente nel tempo.

Dato un determinato numero di sensori IoT distribuiti nello spazio, è possibile costruire un grafo basato sulla similarità spaziale e temporale dei dati raccolti da ciascun sensore. Le anomalie, come i cyberattacchi, possono manifestarsi come cambiamenti inaspettati nelle metriche registrate (ad esempio, nei valori dei sensori). Monitorando l'evoluzione temporale di questi grafi, si possono individuare variazioni che potrebbero indicare attacchi o altre irregolarità nel sistema.

Vorremmo sviluppare un anomaly detector che sfrutti questa correlazione spaziale e temporale dei dati utilizzando una STGNN o una STGAE. Esistono implementazioni di STGNN / STGAE in letteratura, ma andrebbero adattate al nostro specifico contesto. Se il progetto ti interessa, possiamo sentirci per una call di approfondimento.
