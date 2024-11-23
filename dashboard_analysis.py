import os
import json
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Function to load and process files
def load_data(file_path, file_list):
    dataframes = []
    for folder in os.listdir(file_path):
        folder_path = os.path.join(file_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file in file_list:
                full_path = os.path.join(folder_path, file)
                try:
                    with open(full_path, 'r') as f:
                        all_data = json.load(f)
                        for idx, threshold_data in enumerate(all_data, start=1):
                            threshold = idx / 10  # Simulated threshold values 0.1 to 1.0
                            model_data = process_model_data(folder, threshold, threshold_data)
                            dataframes.append(pd.DataFrame(model_data))
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# Helper function to process model data
def process_model_data(folder, threshold, threshold_data):
    model_data = []
    for key, metrics in threshold_data.items():
        if isinstance(metrics, dict):  # Individual class metrics
            model_data.append({
                "Model": folder,
                "Threshold": threshold,
                "Class": key,
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-Score": metrics.get("f1-score"),
                "Support": metrics.get("support"),
                "Metric Type": "Class"
            })
        else:  # Summary metrics (accuracy, macro avg, weighted avg)
            model_data.append({
                "Model": folder,
                "Threshold": threshold,
                "Class": key,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Support": None,
                "Metric Type": key
            })
    return model_data

# Configuration
FILE_PATH = "./new_models/"
FILE_LIST = [
    'corrupted_reports_[1, 2, 3, 4, 5, 6, 7].json',
    'corrupted_reports_[1, 7].json',
    'corrupted_reports_[2, 3].json',
    'corrupted_reports_[3].json'
]

# Load the data
final_df = load_data(FILE_PATH, FILE_LIST)

# Dash Application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Model Performance Dashboard"),
    dcc.Dropdown(
        id="metric",
        options=[
            {"label": "Accuracy", "value": "accuracy"},
            {"label": "Precision", "value": "Precision"},
            {"label": "Recall", "value": "Recall"},
            {"label": "F1-Score", "value": "F1-Score"}
        ],
        value="accuracy",
        placeholder="Select a Metric"
    ),
    dcc.Slider(
        id="threshold-slider",
        min=0.1,
        max=1.0,
        step=0.1,
        value=0.5,
        marks={i / 10: str(i / 10) for i in range(1, 11)}
    ),
    dcc.Dropdown(
        id="sort-field",
        options=[
            {"label": col, "value": col} for col in ["Class", "Precision", "Recall", "F1-Score", "Support"]
        ],
        placeholder="Sort by Field"
    ),
    html.Div([
        html.Label("Filter by Class:"),
        dcc.Input(id="class-filter", type="text", placeholder="Enter class name"),
        html.Label("Filter by Precision (>=):"),
        dcc.Input(id="precision-filter", type="number", placeholder="Enter minimum precision"),
        html.Label("Filter by Recall (>=):"),
        dcc.Input(id="recall-filter", type="number", placeholder="Enter minimum recall"),
    ], style={"margin-bottom": "20px"}),
    dcc.Graph(id="model-performance"),
    html.Div(id="selected-metrics")
])

# Callback to update graph
@app.callback(
    Output("model-performance", "figure"),
    [Input("metric", "value"),
     Input("threshold-slider", "value"),
     Input("sort-field", "value")]
)
def update_graph(selected_metric, threshold, sort_field):
    filtered_df = final_df[final_df["Threshold"] == threshold]
    if sort_field and sort_field in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_field, ascending=True)
    fig = px.bar(
        filtered_df,
        x="Class",
        y=selected_metric,
        title=f"{selected_metric} by Class at Threshold {threshold}",
        labels={"Class": "Class", selected_metric: selected_metric},
        color="Model"  # Added color for better visualization across models
    )
    return fig

# Callback to display filtered table
@app.callback(
    Output("selected-metrics", "children"),
    [Input("threshold-slider", "value"),
     Input("class-filter", "value"),
     Input("precision-filter", "value"),
     Input("recall-filter", "value")]
)
def show_metrics(threshold, class_filter, precision_filter, recall_filter):
    filtered_df = final_df[final_df["Threshold"] == threshold]

    # Apply filters
    if class_filter:
        filtered_df = filtered_df[filtered_df["Class"].str.contains(class_filter, case=False, na=False)]
    if precision_filter is not None:
        filtered_df = filtered_df[filtered_df["Precision"] >= precision_filter]
    if recall_filter is not None:
        filtered_df = filtered_df[filtered_df["Recall"] >= recall_filter]

    if filtered_df.empty:
        return "No data available for the selected filters."
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in filtered_df.columns])),
        html.Tbody([
            html.Tr([html.Td(filtered_df.iloc[i][col]) for col in filtered_df.columns])
            for i in range(len(filtered_df))
        ])
    ])

if __name__ == "__main__":
    app.run_server(debug=True)
