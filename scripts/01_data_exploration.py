import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

sns.set(style="whitegrid")

match_data_path = '../data/raw/port-hueneme-raw-scouter-data.csv'
super_data_path = '../data/raw/port-hueneme-raw-super-data.csv'
output_data_path = '../data/processed/merged_data.csv'
output_plots_path = '../data/results/plots'
output_stats_path = '../data/results/statistics/01_data_exploration_statistics.txt'

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
os.makedirs(output_plots_path, exist_ok=True)
os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

match_df = pd.read_csv(match_data_path)
super_df = pd.read_csv(super_data_path)

match_df['matchNumber'] = match_df['matchNumber'].astype(str)
super_df['matchNumber'] = super_df['matchNumber'].astype(str)
match_df['robotTeam'] = match_df['robotTeam'].astype(str)
super_df['robotTeam'] = super_df['robotTeam'].astype(str)
match_df['robotPosition'] = match_df['robotPosition'].astype(str)
super_df['robotPosition'] = super_df['robotPosition'].astype(str)

merged_df = pd.merge(match_df, super_df, on=['matchNumber', 'robotTeam', 'robotPosition'])
deleted_data_count = len(match_df) + len(super_df) - 2 * len(merged_df)
merged_df.to_csv(output_data_path, index=False)

# Statistics
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

merged_df.fillna(0, inplace=True)

# Plot histograms for key features
key_features = merged_df.columns.difference(['matchNumber', 'robotTeam', 'robotPosition'])
num_plots = len(key_features)
num_cols = 5
num_rows = math.ceil(num_plots / num_cols)
plots_per_figure = 20

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
    plt.close()

# Correlation analysis
numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(20, 16))
correlation_matrix = merged_df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix - Merged Data')
plt.savefig(os.path.join(output_plots_path, 'correlation_matrix.png'))
plt.close()

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

print("Data exploration complete. Merged data saved to:", output_data_path)
print("Statistics saved to:", output_stats_path)