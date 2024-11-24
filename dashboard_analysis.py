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
                        for idx, threshold_data in enumerate(all_data, start=0):
                            threshold = idx / 10  # Simulated threshold values 0.1 to 1.0
                            model_data = process_model_data(folder, file, threshold, threshold_data)
                            dataframes.append(pd.DataFrame(model_data))
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# Helper function to process model data
def process_model_data(folder, file, threshold, threshold_data):
    model_data = []
    for key, metrics in threshold_data.items():
        if isinstance(metrics, dict):  # Individual class metrics
            model_data.append({
                "Model": folder,
                "File": file,
                "Threshold": threshold,
                "Class": key,
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-Score": metrics.get("f1-score"),
                "Accuracy": None,
                "Support": metrics.get("support"),
                "Metric Type": "Class"
            })
        else:  # Summary metrics (accuracy, macro avg, weighted avg)
            model_data.append({
                "Model": folder,
                "File": file,
                "Threshold": threshold,
                "Class": key,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Accuracy": metrics if key == "accuracy" else None,
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
#print(final_df[(final_df["Model"] == "temporal_sandwich_model_with_gate_data_shuffled_edge_weight") & (final_df["File"] == "corrupted_reports_[3].json")])

# Dash Application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Model Comparison Dashboard"),
    html.Div([
        html.H2("Anomaly Detection on Time Series Graph"),
        html.Label("Select Baseline Model:"),
        dcc.Dropdown(
            id="baseline-model",
            options=[{"label": model, "value": model} for model in final_df["Model"].unique()],
            placeholder="Select Baseline Model"
        ),
        html.Label("Select New Model:"),
        dcc.Dropdown(
            id="new-model",
            options=[{"label": model, "value": model} for model in final_df["Model"].unique()],
            placeholder="Select New Model"
        ),
        html.Label("Select File:"),
        dcc.Dropdown(
            id="file-filter",
            placeholder="Select File (common to both models)"
        )
    ], style={"margin-bottom": "20px"}),
    html.Label("Select Corruption probability:"),
    dcc.Slider(
        id="threshold-slider",
        min=0.0,
        max=1.0,
        step=0.1,
        value=0.5,
        marks={i / 10: str(i / 10) for i in range(1, 11)}
    ),
    html.Div(id="tables-row", style={"display": "flex", "justify-content": "space-between"})
])

# Callback to update file options based on selected models
@app.callback(
    Output("file-filter", "options"),
    [Input("baseline-model", "value"),
     Input("new-model", "value")]
)
def update_file_options(baseline, new):
    if not baseline or not new:
        return []
    baseline_files = final_df[final_df["Model"] == baseline]["File"].unique()
    new_files = final_df[final_df["Model"] == new]["File"].unique()
    common_files = list(set(baseline_files) & set(new_files))
    return [{"label": file, "value": file} for file in common_files]

# Callback to show tables
@app.callback(
    Output("tables-row", "children"),
    [Input("baseline-model", "value"),
     Input("new-model", "value"),
     Input("file-filter", "value"),
     Input("threshold-slider", "value")]
)
def update_tables(baseline, new, file, threshold):
    if not baseline or not new or not file:
        return "Please select both models and a file."
    
    # Filter data for the selected models and file
    baseline_data = final_df[
        (final_df["Model"] == baseline) & 
        (final_df["File"] == file) & 
        (final_df["Threshold"] == threshold)
    ]
    new_data = final_df[
        (final_df["Model"] == new) & 
        (final_df["File"] == file) & 
        (final_df["Threshold"] == threshold)
    ]

    # Compute average metrics for each model across all thresholds
    avg_baseline = final_df[
        (final_df["Model"] == baseline) & 
        (final_df["File"] == file)
    ].groupby("Class").mean(numeric_only=True).reset_index()

    avg_new = final_df[
        (final_df["Model"] == new) & 
        (final_df["File"] == file)
    ].groupby("Class").mean(numeric_only=True).reset_index()

    # Merge for difference calculation
    comparison_df = pd.merge(
        baseline_data,
        new_data,
        on=["Class", "Threshold"],
        suffixes=("_baseline", "_new")
    )
    comparison_df["Precision_Diff"] = comparison_df["Precision_new"] - comparison_df["Precision_baseline"]
    comparison_df["Recall_Diff"] = comparison_df["Recall_new"] - comparison_df["Recall_baseline"]
    comparison_df["F1-Score_Diff"] = comparison_df["F1-Score_new"] - comparison_df["F1-Score_baseline"]
    comparison_df["Accuracy_Diff"] = comparison_df["Accuracy_new"] - comparison_df["Accuracy_baseline"]
    
    avg_comparison_df = pd.merge(
        avg_baseline,
        avg_new,
        on="Class",
        suffixes=("_baseline", "_new")
    )
    
    avg_comparison_df["Precision_Diff"] = avg_comparison_df["Precision_new"] - avg_comparison_df["Precision_baseline"]
    avg_comparison_df["Recall_Diff"] = avg_comparison_df["Recall_new"] - avg_comparison_df["Recall_baseline"]
    avg_comparison_df["F1-Score_Diff"] = avg_comparison_df["F1-Score_new"] - avg_comparison_df["F1-Score_baseline"]
    avg_comparison_df["Accuracy_Diff"] = avg_comparison_df["Accuracy_new"] - avg_comparison_df["Accuracy_baseline"]

    def create_table(df, columns, title):
        return html.Div([
            html.H4(title),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in columns])),
                html.Tbody([
                    html.Tr([html.Td(f'{df.iloc[i][col]:.2f}' if isinstance(df.iloc[i][col], (float, int)) else df.iloc[i][col]) for col in columns])
                    for i in range(len(df))
                ])
            ], style={"margin": "10px"})
        ], style={"flex": "1"})

    def create_difference_table(df):
        return html.Div([
            html.H4("Differences Table"),
            html.Table([
            html.Thead(html.Tr([html.Th(col) for col in ["Class", "Precision_Diff", "Recall_Diff", "Accuracy", "F1-Score_Diff"]])),
            html.Tbody([
                html.Tr([
            html.Td(df.iloc[i]["Class"]),
            html.Td(html.Span(f'{df.iloc[i]["Precision_Diff"]:.2f}', style={"color": "green" if round(df.iloc[i]["Precision_Diff"], 2) > 0 else "red" if round(df.iloc[i]["Precision_Diff"], 2) < 0 else "grey"})),
            html.Td(html.Span(f'{df.iloc[i]["Recall_Diff"]:.2f}', style={"color": "green" if round(df.iloc[i]["Recall_Diff"], 2) > 0 else "red" if round(df.iloc[i]["Recall_Diff"], 2) < 0 else "grey"})),
            html.Td(html.Span(f'{df.iloc[i]["Accuracy_Diff"]:.2f}', style={"color": "green" if round(df.iloc[i]["Accuracy_Diff"], 2) > 0 else "red" if round(df.iloc[i]["Accuracy_Diff"], 2) < 0 else "grey"})),
            html.Td(html.Span(f'{df.iloc[i]["F1-Score_Diff"]:.2f}', style={"color": "green" if round(df.iloc[i]["F1-Score_Diff"], 2) > 0 else "red" if round(df.iloc[i]["F1-Score_Diff"], 2) < 0 else "grey"})),
                ])
                for i in range(len(df))
            ])
            ], style={"margin": "10px"})
        ], style={"flex": "1"})

    # Create tables for threshold-specific and average metrics
    baseline_table = create_table(baseline_data, ["Class", "Precision", "Recall", "Accuracy", "F1-Score"], "Baseline Model Table")
    new_model_table = create_table(new_data, ["Class", "Precision", "Recall", "Accuracy", "F1-Score"], "New Model Table")
    diff_table = create_difference_table(comparison_df)
    avg_baseline_table = create_table(avg_baseline, ["Class", "Precision", "Recall", "Accuracy", "F1-Score"], "Baseline Average Metrics")
    avg_new_table = create_table(avg_new, ["Class", "Precision", "Recall", "Accuracy", "F1-Score"], "New Model Average Metrics")
    avg_diff_table = create_difference_table(avg_comparison_df)

    # Arrange the layout: two rows
    return html.Div([
        html.Div([
            baseline_table,
            new_model_table,
            diff_table
        ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"}),
        html.Div([
            avg_baseline_table,
            avg_new_table,
            avg_diff_table
        ], style={"display": "flex", "justify-content": "space-between"})
    ])

if __name__ == "__main__":
    app.run_server(debug=True)