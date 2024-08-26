import os
import shutil
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Define the paths to the results and processed directories
results_dir = os.path.join("data", "results")
processed_dir = os.path.join("data", "processed")

# Delete the results folder and its contents
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
    print(f"Deleted and cleared: {results_dir}")
    os.makedirs(results_dir)  # Recreate the results folder

# Delete the processed folder and its contents
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
    print(f"Deleted and cleared: {processed_dir}")
    os.makedirs(processed_dir)  # Recreate the processed folder

print("All output directories have been cleared and recreated.")
