import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Caricamento dati da file
def load_data():
    all_data = []
    file_path = "data/subdata_extended_augmented/"  # Sostituisci con il percorso della tua cartella
    for filename in os.listdir(file_path):
        if filename.endswith(".csv"):
            data = pd.read_csv(os.path.join(file_path, filename))
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

data = load_data()
data['year'] = data['year'].astype(int)
data['month'] = data['month'].astype(int)
data['day'] = data['day'].astype(int)
data['hour'] = data['hour'].astype(int)
data['minute'] = data['minute'].astype(int)
data['second'] = data['second'].astype(int)
data['atrack'] = data['atrack'].astype(int)
data['xtrack'] = data['atrack'].astype(int)


# Inizializzazione dell'app Dash
app = dash.Dash(__name__)
app.layout = html.Div([  
    html.H1("Dashboard Analisi Dati Giornalieri"),
    
    # Filtro per selezionare i campi da visualizzare
    html.Label("Seleziona il campo da analizzare"),
    dcc.Dropdown(
        id='field-dropdown',
        options=[{'label': col, 'value': col} for col in data.columns if col not in ['lat', 'lon', 'year', 'month', 'day', 'hour', 'minute', 'second', 'atrack', 'xtrack']],
        value='surf_temp_masked',
        clearable=False
    ),

    # Slider temporale
    html.Label("Seleziona la data"),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # Intervallo per l'animazione
    dcc.Slider(
        id='date-slider',
        min=datetime(data['year'].min(), data['month'].min(), data['day'].min()).timestamp(),
        max=datetime(data['year'].max(), data['month'].max(), data['day'].max()).timestamp(),
        value=datetime(data['year'].min(), data['month'].min(), data['day'].min()).timestamp(),
        marks={datetime(year, month, day).timestamp(): f"{day}-{month}-{year}" 
               for year, month, day in zip(data['year'], data['month'], data['day'])},
        step=24*60*60
    ),
    html.Button('Play', id='play-button', n_clicks=0),
    html.Button('Pause', id='pause-button', n_clicks=0),

    # Grafico mappa per la visualizzazione spaziale
    dcc.Graph(id="map-graph"),

    # Grafico temporale per visualizzazione di trend nel tempo
    dcc.Graph(id="time-series-graph")
])

# Variabile per gestire l'animazione
animation_playing = False

# Callback per controllare l'animazione
@app.callback(
    Output('interval-component', 'disabled'),
    Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks')
)
def toggle_animation(play_clicks, pause_clicks):
    global animation_playing
    if play_clicks > pause_clicks:
        animation_playing = True
        return False  # Abilita l'intervallo per iniziare l'animazione
    else:
        animation_playing = False
        return True  # Disabilita l'intervallo per fermare l'animazione

# Callback per avanzare lo slider automaticamente
@app.callback(
    Output('date-slider', 'value'),
    Input('interval-component', 'n_intervals'),
    Input('date-slider', 'value')
)
def animate_slider(n_intervals, current_value):
    if animation_playing:
        # Aumenta il valore dello slider per far avanzare la data di un giorno
        next_value = current_value + 24*60*60
        # Ricomincia dall'inizio se supera il valore massimo
        if next_value > datetime(data['year'].max(), data['month'].max(), data['day'].max()).timestamp():
            next_value = datetime(data['year'].min(), data['month'].min(), data['day'].min()).timestamp()
        return next_value
    return current_value


# Callback per aggiornare la mappa
@app.callback(
    Output('map-graph', 'figure'),
    Input('date-slider', 'value'),
    Input('field-dropdown', 'value')
)
def update_map(selected_date, selected_field):
    # Filtrare i dati in base alla data selezionata
    selected_date = datetime.fromtimestamp(selected_date)
    filtered_data = data[(data['year'] == selected_date.year) & 
                         (data['month'] == selected_date.month) & 
                         (data['day'] == selected_date.day)]
    
    fig = px.scatter_mapbox(filtered_data, lat="lat", lon="lon", color=selected_field,
                            title=f"Distribuzione spaziale di {selected_field}",
                            mapbox_style="open-street-map", zoom=2)
    return fig

# Callback per aggiornare il grafico temporale
@app.callback(
    Output('time-series-graph', 'figure'),
    Input('field-dropdown', 'value')
)
def update_time_series(selected_field):
    # Aggregazione dati giornalieri
    daily_data = data.groupby(['year', 'month', 'day'])[selected_field].mean().reset_index()
    daily_data['date'] = pd.to_datetime(daily_data[['year', 'month', 'day']])
    
    fig = px.line(daily_data, x="date", y=selected_field,
                  title=f"Trend giornaliero di {selected_field} nel tempo")
    return fig

# Esecuzione dell'app
if __name__ == '__main__':
    app.run_server(debug=True)
