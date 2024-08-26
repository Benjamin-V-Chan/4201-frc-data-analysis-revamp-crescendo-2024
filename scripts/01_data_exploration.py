import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Access the correct paths
match_data_path = config['data_paths']['raw_data']['scouter_data']
super_data_path = config['data_paths']['raw_data']['super_data']
output_data_path = config['data_paths']['processed_data']['merged_data']
output_plots_path = config['data_paths']['results']['plots']
output_stats_path = config['data_paths']['results']['statistics']['data_exploration']

# Create necessary directories if they don't exist
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
os.makedirs(output_plots_path, exist_ok=True)
os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

# Load data into DataFrames
match_df = pd.read_csv(match_data_path)
super_df = pd.read_csv(super_data_path)

# Ensure 'matchNumber', 'robotTeam', and 'robotPosition' columns are of the same data type in both DataFrames
match_df['matchNumber'] = match_df['matchNumber'].astype(str)
super_df['matchNumber'] = super_df['matchNumber'].astype(str)
match_df['robotTeam'] = match_df['robotTeam'].astype(str)
super_df['robotTeam'] = super_df['robotTeam'].astype(str)
match_df['robotPosition'] = match_df['robotPosition'].astype(str)
super_df['robotPosition'] = super_df['robotPosition'].astype(str)

# Merge the DataFrames on 'matchNumber', 'robotTeam', and 'robotPosition' columns
merged_df = pd.merge(match_df, super_df, on=['matchNumber', 'robotTeam', 'robotPosition'])

# Calculate the number of rows deleted due to non-matching entries
deleted_data_count = len(match_df) + len(super_df) - 2 * len(merged_df)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_data_path, index=False)

# Save statistics and numerical information to a text file
with open(output_stats_path, 'w') as f:
    f.write(f"Match Data Shape: {match_df.shape}\n")
    f.write(f"Super Data Shape: {super_df.shape}\n")
    f.write(f"Merged Data Shape: {merged_df.shape}\n")
    f.write(f"Number of rows deleted due to non-matching entries: {deleted_data_count}\n\n")

    f.write("Match Data Info:\n")
    match_df.info(buf=f)
    f.write("\n\nSuper Data Info:\n")
    super_df.info(buf=f)
    f.write("\n\nMerged Data Info:\n")
    merged_df.info(buf=f)

    f.write("\n\nMatch Data Statistics:\n")
    f.write(match_df.describe().to_string())
    f.write("\n\nSuper Data Statistics:\n")
    f.write(super_df.describe().to_string())
    f.write("\n\nMerged Data Statistics:\n")
    f.write(merged_df.describe().to_string())

    f.write("\n\nFirst few rows of match data:\n")
    f.write(match_df.head().to_string())
    f.write("\n\nFirst few rows of super data:\n")
    f.write(super_df.head().to_string())
    f.write("\n\nFirst few rows of merged data:\n")
    f.write(merged_df.head().to_string())

    missing_values = merged_df.isnull().sum()
    f.write("\n\nMissing Values:\n")
    f.write(missing_values.to_string())

# Handle missing values (for simplicity, we'll fill with 0)
merged_df.fillna(0, inplace=True)

# Identify key features for distribution plots
key_features = merged_df.columns.difference(['matchNumber', 'robotTeam', 'robotPosition'])

# Determine the number of rows and columns for subplots
num_plots = len(key_features)
num_cols = 5
num_rows = math.ceil(num_plots / num_cols)
plots_per_figure = 20  # Number of plots per figure

# Plot histograms for key features
for start in range(0, num_plots, plots_per_figure):
    end = min(start + plots_per_figure, num_plots)
    features = key_features[start:end]
    num_features = len(features)
    rows = math.ceil(num_features / num_cols)
    
    plt.figure(figsize=(num_cols * 4, rows * 4))
    for i, feature in enumerate(features, 1):
        plt.subplot(rows, num_cols, i)
        sns.histplot(merged_df[feature], bins=30, kde=True)
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_path, f'feature_distributions_{start // plots_per_figure + 1}.png'))
    plt.close()  # Close the figure to avoid overlap

# Filter numeric columns for correlation matrix
numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns

# Correlation analysis
plt.figure(figsize=(20, 16))
correlation_matrix = merged_df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix - Merged Data')
plt.savefig(os.path.join(output_plots_path, 'correlation_matrix.png'))
plt.close()  # Close the figure to avoid overlap

# Side-by-side box plots for auto distances
auto_distances = ['autoNotes.near', 'autoNotes.mid', 'autoNotes.far']
plt.figure(figsize=(12, 8))
sns.boxplot(data=merged_df[auto_distances])
plt.title('Side-by-Side Box Plots for Auto Distances')
plt.xlabel('Auto Distances')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig(os.path.join(output_plots_path, 'auto_distances_boxplots.png'))
plt.close()

# Side-by-side box plots for tele distances
tele_distances = ['teleNotes.near', 'teleNotes.mid', 'teleNotes.far']
plt.figure(figsize=(12, 8))
sns.boxplot(data=merged_df[tele_distances])
plt.title('Side-by-Side Box Plots for Tele Distances')
plt.xlabel('Tele Distances')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig(os.path.join(output_plots_path, 'tele_distances_boxplots.png'))
plt.close()

print(f"Data exploration complete. Merged data saved to: {output_data_path}")
print(f"Statistics saved to: {output_stats_path}")
print(f"Plots saved to: {output_plots_path}")
