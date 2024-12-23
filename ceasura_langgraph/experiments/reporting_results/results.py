import pandas as pd

# Read data from CSV
file_path = "data.csv" 
df = pd.read_csv(file_path, sep='\t')



print(df.columns)

# Convert relevant columns to float
float_columns = ["ours_Accuracy", "ours_Steps", "ours_Latency", "ours_Total_Tokens","ours_Cost","caesura_Accuracy","caesura_Steps",
                 "caesura_Latency","caesura_Total_Tokens","caesura_Cost"]
df[float_columns] = df[float_columns].replace(",", ".", regex=True)

df[float_columns] = df[float_columns].astype(float)
# Group by Output Type and Modality
grouped = df.groupby(["Output_Type", "Modality"]).agg({
    "ours_Accuracy": "sum",
    "ours_Steps": ["mean", "min", "max"],
    "ours_Latency": ["mean", "min", "max"],
    "ours_Total_Tokens": ["mean", "min", "max"],
    "ours_Cost": "sum",
    "caesura_Accuracy": "sum",
    "caesura_Steps": ["mean", "min", "max"],
    "caesura_Latency": ["mean", "min", "max"],
    "caesura_Total_Tokens":["mean", "min", "max"],
    "caesura_Cost": "sum",
}).reset_index()

# Flatten MultiIndex columns
grouped.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns
]

# Add a column for total counts in each group
group_counts = df.groupby(["Output_Type", "Modality"]).size().reset_index(name="Total_Count")
grouped = grouped.merge(group_counts, on=["Output_Type", "Modality"])



# Calculate accuracy as sum / total
grouped["ours_Avg_Accuracy"] = grouped["ours_Accuracy_sum"] / grouped["Total_Count"]
grouped["caesura_Avg_Accuracy"] = grouped["caesura_Accuracy_sum"] / grouped["Total_Count"]

# Clean up column names
grouped.rename(columns={
    "ours_Steps_mean": "ours_Avg_Steps", "ours_Steps_min": "ours_Min_Steps", "ours_Steps_max": "ours_Max_Steps",
    "ours_Latency_mean": "ours_Avg_Latency", "ours_Latency_min": "ours_Min_Latency", "ours_Latency_max": "ours_Max_Latency",
    "ours_Total_Tokens_sum": "ours_Sum_Total_Tokens", "ours_Total_Tokens_min": "ours_Min_Total_Tokens", 
    "ours_Total_Tokens_max": "ours_Max_Total_Tokens",
    "caesura__Steps_mean": "caesura_Avg_Steps", "caesura_Steps_min": "caesura_Min_Steps", "caesura_Steps_max": "caesura_Max_Steps",
    "ours_Latency_mean": "ours_Avg_Latency", "ours_Latency_min": "ours_Min_Latency", "ours_Latency_max": "ours_Max_Latency",
    "caesura_Total_Tokens_sum": "caesura_Sum_Total_Tokens", "caesura_Total_Tokens_min": "caesura_Min_Total_Tokens", 
    "caesura_Total_Tokens_max": "caesura_Max_Total_Tokens"
}, inplace=True)

print(grouped)
# Save the result to a CSV file
output_file = "grouped_results.csv"
grouped.to_csv(output_file, index=False)
print(f"Grouped results saved to {output_file}")