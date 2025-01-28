import pandas as pd
import os
import json
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Paths from config
input_data_path = config['data_paths']['processed_data']['cleaned_data']
output_team_stats_path = config['data_paths']['results']['statistics']['team_statistics']
photos_path = config['data_paths']['raw_data']['photos']

# Create necessary directories if they don't exist
os.makedirs(os.path.dirname(output_team_stats_path), exist_ok=True)

# Load cleaned data into a DataFrame
df = pd.read_csv(input_data_path)

# Calculate match-level statistics
df['totalAutoNotes'] = df[['autoNotes.near', 'autoNotes.mid', 'autoNotes.far', 'autoNotes.amp']].sum(axis=1)
df['totalTeleNotes'] = df[['teleNotes.near', 'teleNotes.mid', 'teleNotes.far', 'teleNotes.amp']].sum(axis=1)
df['totalMatchNotes'] = df['totalAutoNotes'] + df['totalTeleNotes'] + df['trapNotes']
df['totalFouls'] = df[['podiumFoul', 'zoneFoul', 'stageFoul', 'overExtChute']].sum(axis=1)

# Group by team number
team_stats = df.groupby('robotTeam').agg(
    numMatches=('matchNumber', 'count'),
    meanAutoNotes=('totalAutoNotes', 'mean'),
    medianAutoNotes=('totalAutoNotes', 'median'),
    minAutoNotes=('totalAutoNotes', 'min'),
    maxAutoNotes=('totalAutoNotes', 'max'),
    stdAutoNotes=('totalAutoNotes', 'std'),
    meanTeleNotes=('totalTeleNotes', 'mean'),
    medianTeleNotes=('totalTeleNotes', 'median'),
    minTeleNotes=('totalTeleNotes', 'min'),
    maxTeleNotes=('totalTeleNotes', 'max'),
    stdTeleNotes=('totalTeleNotes', 'std'),
    meanFouls=('totalFouls', 'mean'),
    medianFouls=('totalFouls', 'median'),
    minFouls=('totalFouls', 'min'),
    maxFouls=('totalFouls', 'max'),
    stdFouls=('totalFouls', 'std'),
    noDefPercent=('defense', lambda x: (x == 'noDef').mean()),
    someDefPercent=('defense', lambda x: (x == 'someDef').mean()),
    fullDefPercent=('defense', lambda x: (x == 'fullDef').mean()),
    climbNonePercent=('climb', lambda x: (x == 'None').mean()),
    climbLowPercent=('climb', lambda x: (x == 'Low').mean()),
    climbMidPercent=('climb', lambda x: (x == 'Mid').mean()),
    climbHighPercent=('climb', lambda x: (x == 'High').mean())
)

# Calculate additional statistics
team_stats['mainDefType'] = team_stats[['noDefPercent', 'someDefPercent', 'fullDefPercent']].idxmax(axis=1)
team_stats['mainClimbType'] = team_stats[['climbNonePercent', 'climbLowPercent', 'climbMidPercent', 'climbHighPercent']].idxmax(axis=1)

# Percent calculations for fouls and autonotes
team_stats['percentNoFouls'] = df.groupby('robotTeam')['totalFouls'].apply(lambda x: (x == 0).mean())
team_stats['percentOneFoul'] = df.groupby('robotTeam')['totalFouls'].apply(lambda x: (x >= 1).mean())
team_stats['percentThreeFouls'] = df.groupby('robotTeam')['totalFouls'].apply(lambda x: (x >= 3).mean())
team_stats['percentFiveFouls'] = df.groupby('robotTeam')['totalFouls'].apply(lambda x: (x >= 5).mean())
team_stats['percentEightFouls'] = df.groupby('robotTeam')['totalFouls'].apply(lambda x: (x >= 8).mean())

team_stats['percentAutoNotes1'] = df.groupby('robotTeam')['totalAutoNotes'].apply(lambda x: (x >= 1).mean())
team_stats['percentAutoNotes3'] = df.groupby('robotTeam')['totalAutoNotes'].apply(lambda x: (x >= 3).mean())
team_stats['percentAutoNotes5'] = df.groupby('robotTeam')['totalAutoNotes'].apply(lambda x: (x >= 5).mean())

# Count of matches reaching max and second max autonotes
team_stats['numMaxAutoNotes'] = df.groupby('robotTeam')['totalAutoNotes'].apply(lambda x: (x == x.max()).sum())
team_stats['numSecondMaxAutoNotes'] = df.groupby('robotTeam')['totalAutoNotes'].apply(lambda x: (x == x.nlargest(2).iloc[-1]).sum())

# Add photos based on robot team
team_stats['teamPhoto'] = team_stats.index.map(lambda x: os.path.join(photos_path, f"{x}.jpg"))

# Clustering Analysis
features_for_clustering = ['meanAutoNotes', 'meanTeleNotes', 'meanFouls', 'numMatches']
X = team_stats[features_for_clustering].fillna(0)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
team_stats['kmeans_cluster'] = kmeans.fit_predict(X)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
team_stats['agg_cluster'] = agg_clustering.fit_predict(X)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
team_stats['dbscan_cluster'] = dbscan.fit_predict(X)

# Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
team_stats['gmm_cluster'] = gmm.fit_predict(X)

# Mean Shift Clustering
mean_shift = MeanShift()
team_stats['mean_shift_cluster'] = mean_shift.fit_predict(X)

# Save the team statistics to a CSV file
team_stats.reset_index().to_csv(output_team_stats_path, index=False)

print("Team statistics calculation and clustering complete. Data saved to:", output_team_stats_path)
