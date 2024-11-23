import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import os

file_list = ['corrupted_reports_[1, 2, 3, 4, 5, 6, 7].json',
             'corrupted_reports_[1, 7].json ',
             'corrupted_reports_[2, 3].json' ,
             'corrupted_reports_[3].json'
             ]

# File paths for uploaded file simulation
file_path = "/new_models/"


# Reading the file content into a structured dataframe
dataframes = []

# Itereate thoug all the folders of the directory
for root, dirs, files in os.walk(file_path):
    for file in files:
        if file in file_list:
            print(os.path.join(root, file))

            with open(os.path.join(root, file), 'r') as f:
                all_data = json.load(f)
                for idx, threshold_data in enumerate(all_data, start=1):
                    # Convert the threshold data into a dataframe-friendly format
                    model_data = []
                    for key, metrics in threshold_data.items():
                        if isinstance(metrics, dict):  # Individual class metrics
                            model_data.append({
                                "Threshold": idx / 10,  # Simulated threshold values 0.1 to 1.0
                                "Class": key,
                                "Precision": metrics.get("precision", None),
                                "Recall": metrics.get("recall", None),
                                "F1-Score": metrics.get("f1-score", None),
                                "Support": metrics.get("support", None),
                                "Metric Type": "Class"
                            })
                        else:
                            model_data.append({
                                "Threshold": idx / 10,
                                "Class": key,
                                "Precision": None,
                                "Recall": None,
                                "F1-Score": None,
                                "Support": None,
                                "Metric Type": key
                            })
                    dataframes.append(pd.DataFrame(model_data))

# Concatenate all thresholds into a single dataframe
final_df = pd.concat(dataframes, ignore_index=True)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Model Performance Dashboard"),
    dcc.Dropdown(
        id="metric",
        options=[
            {"label": "Accuracy", "value": "accuracy"},
            {"label": "Precision", "value": "precision"},
            {"label": "Recall", "value": "recall"},
            {"label": "F1-Score", "value": "f1"}
        ],
        value="accuracy"
    ),
    dcc.Graph(id="model-performance"),
    dcc.Slider(
        id="threshold-slider",
        min=0,
        max=1,
        step=0.1,
        value=0.5,
        marks={i/10: str(i/10) for i in range(11)}
    ),
    html.Div(id="selected-metrics")
])

@app.callback(
    Output("model-performance", "figure"),
    Input("metric", "value"),
    Input("threshold-slider", "value")
)
def update_graph(selected_metric, threshold):
    filtered_df = df[df["threshold"] == threshold]
    fig = px.bar(
        filtered_df,
        x="model",
        y=selected_metric,
        title=f"{selected_metric.capitalize()} by Model at Threshold {threshold}",
        labels={"model": "Model", selected_metric: selected_metric.capitalize()}
    )
    return fig

@app.callback(
    Output("selected-metrics", "children"),
    Input("threshold-slider", "value")
)
def show_metrics(threshold):
    filtered_df = df[df["threshold"] == threshold]
    return html.Table([
        html.Tr([html.Th(col) for col in filtered_df.columns]),
        *[html.Tr([html.Td(filtered_df.iloc[i][col]) for col in filtered_df.columns]) for i in range(len(filtered_df))]
    ])

if __name__ == "__main__":
    app.run_server(debug=True)
