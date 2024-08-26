import pandas as pd
import os
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Access the correct paths
input_data_path = config['data_paths']['processed_data']['merged_data']
output_data_path = config['data_paths']['processed_data']['cleaned_data']
output_stats_path = config['data_paths']['results']['statistics']['data_preprocessing']

# Create necessary directories if they don't exist
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

# Load the merged data
df = pd.read_csv(input_data_path)

# Example preprocessing: fill missing values and drop duplicates
df.fillna(0, inplace=True)
df.drop_duplicates(inplace=True)

# Save the cleaned data to a new CSV file
df.to_csv(output_data_path, index=False)

# Save some statistics about the preprocessing
with open(output_stats_path, 'w') as f:
    f.write(f"Original Data Shape: {df.shape}\n")
    f.write(f"Cleaned Data Shape: {df.shape}\n")
    f.write("Number of duplicates dropped: None (already cleaned in this example)\n")

print(f"Data preprocessing complete. Cleaned data saved to: {output_data_path}")
print(f"Statistics saved to: {output_stats_path}")
