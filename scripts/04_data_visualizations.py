import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Paths from config
input_team_stats_path = config['data_paths']['results']['statistics']['team_statistics']
output_plots_path = config['data_paths']['results']['plots']

# Create necessary directories if they don't exist
os.makedirs(output_plots_path, exist_ok=True)

# Load team statistics into a DataFrame
team_stats = pd.read_csv(input_team_stats_path)

# Set plotting style
sns.set(style="whitegrid")

# Plot: Distribution of Mean Auto Notes Across Teams
plt.figure(figsize=(10, 6))
sns.histplot(team_stats['meanAutoNotes'], kde=True)
plt.title('Distribution of Mean Auto Notes Across Teams')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_plots_path, 'mean_auto_notes_distribution.png'))
plt.close()

# Plot: Distribution of Mean Tele Notes Across Teams
plt.figure(figsize=(10, 6))
sns.histplot(team_stats['meanTeleNotes'], kde=True)
plt.title('Distribution of Mean Tele Notes Across Teams')
plt.xlabel('Mean Tele Notes')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_plots_path, 'mean_tele_notes_distribution.png'))
plt.close()

# Plot: Distribution of Mean Fouls Across Teams
plt.figure(figsize=(10, 6))
sns.histplot(team_stats['meanFouls'], kde=True)
plt.title('Distribution of Mean Fouls Across Teams')
plt.xlabel('Mean Fouls')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_plots_path, 'mean_fouls_distribution.png'))
plt.close()

# Plot: Mean Auto Notes vs. Mean Tele Notes (K-Means)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(team_stats['meanAutoNotes'], team_stats['meanTeleNotes'], c=team_stats['kmeans_cluster'], cmap='viridis', s=100)
plt.title('K-Means Clusters Based on Auto and Tele Notes')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Mean Tele Notes')
plt.colorbar(scatter, label='Cluster')  # Attach the colorbar to the scatter plot
plt.savefig(os.path.join(output_plots_path, 'kmeans_clusters.png'))
plt.close()

# Plot: Mean Auto Notes vs. Mean Tele Notes (Agglomerative)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(team_stats['meanAutoNotes'], team_stats['meanTeleNotes'], c=team_stats['agg_cluster'], cmap='plasma', s=100)
plt.title('Agglomerative Clusters Based on Auto and Tele Notes')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Mean Tele Notes')
plt.colorbar(scatter, label='Cluster')  # Attach the colorbar to the scatter plot
plt.savefig(os.path.join(output_plots_path, 'agg_clusters.png'))
plt.close()

# Plot: Mean Auto Notes vs. Mean Tele Notes (DBSCAN)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(team_stats['meanAutoNotes'], team_stats['meanTeleNotes'], c=team_stats['dbscan_cluster'], cmap='coolwarm', s=100)
plt.title('DBSCAN Clusters Based on Auto and Tele Notes')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Mean Tele Notes')
plt.colorbar(scatter, label='Cluster')  # Attach the colorbar to the scatter plot
plt.savefig(os.path.join(output_plots_path, 'dbscan_clusters.png'))
plt.close()

# Plot: Mean Auto Notes vs. Mean Tele Notes (GMM)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(team_stats['meanAutoNotes'], team_stats['meanTeleNotes'], c=team_stats['gmm_cluster'], cmap='cividis', s=100)
plt.title('GMM Clusters Based on Auto and Tele Notes')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Mean Tele Notes')
plt.colorbar(scatter, label='Cluster')  # Attach the colorbar to the scatter plot
plt.savefig(os.path.join(output_plots_path, 'gmm_clusters.png'))
plt.close()

# Plot: Mean Auto Notes vs. Mean Tele Notes (Mean Shift)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(team_stats['meanAutoNotes'], team_stats['meanTeleNotes'], c=team_stats['mean_shift_cluster'], cmap='magma', s=100)
plt.title('Mean Shift Clusters Based on Auto and Tele Notes')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Mean Tele Notes')
plt.colorbar(scatter, label='Cluster')  # Attach the colorbar to the scatter plot
plt.savefig(os.path.join(output_plots_path, 'mean_shift_clusters.png'))
plt.close()

# Plot: Mean Auto Notes vs. Mean Fouls
plt.figure(figsize=(10, 8))
sns.scatterplot(data=team_stats, x='meanAutoNotes', y='meanFouls', hue='kmeans_cluster', palette='viridis', s=100)
plt.title('Mean Auto Notes vs. Mean Fouls')
plt.xlabel('Mean Auto Notes')
plt.ylabel('Mean Fouls')
plt.savefig(os.path.join(output_plots_path, 'mean_auto_vs_fouls.png'))
plt.close()

# Plot: Mean Tele Notes vs. Mean Fouls
plt.figure(figsize=(10, 8))
sns.scatterplot(data=team_stats, x='meanTeleNotes', y='meanFouls', hue='kmeans_cluster', palette='viridis', s=100)
plt.title('Mean Tele Notes vs. Mean Fouls')
plt.xlabel('Mean Tele Notes')
plt.ylabel('Mean Fouls')
plt.savefig(os.path.join(output_plots_path, 'mean_tele_vs_fouls.png'))
plt.close()

# Plot: Percent No Def vs. Percent Full Def
plt.figure(figsize=(10, 8))
sns.scatterplot(data=team_stats, x='noDefPercent', y='fullDefPercent', hue='kmeans_cluster', palette='viridis', s=100)
plt.title('Percent No Defense vs. Percent Full Defense')
plt.xlabel('Percent No Defense')
plt.ylabel('Percent Full Defense')
plt.savefig(os.path.join(output_plots_path, 'no_def_vs_full_def.png'))
plt.close()

# Plot: Percent No Fouls vs. Mean Auto Notes
plt.figure(figsize=(10, 8))
sns.scatterplot(data=team_stats, x='percentNoFouls', y='meanAutoNotes', hue='kmeans_cluster', palette='viridis', s=100)
plt.title('Percent No Fouls vs. Mean Auto Notes')
plt.xlabel('Percent No Fouls')
plt.ylabel('Mean Auto Notes')
plt.savefig(os.path.join(output_plots_path, 'no_fouls_vs_auto_notes.png'))
plt.close()

# Heatmap: Correlation Matrix of Team Statistics
plt.figure(figsize=(12, 10))
correlation_matrix = team_stats.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Team Statistics')
plt.savefig(os.path.join(output_plots_path, 'correlation_matrix_team_stats.png'))
plt.close()

print("Visualizations complete. Plots saved to:", output_plots_path)
