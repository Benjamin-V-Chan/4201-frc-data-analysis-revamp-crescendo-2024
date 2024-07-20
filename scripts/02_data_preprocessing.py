import pandas as pd
import os

input_data_path = '../data/processed/merged_data.csv'
output_data_path = '../data/processed/cleaned_data.csv'
output_stats_path = '../data/results/statistics/02_data_preprocessing_statistics.txt'

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

merged_df = pd.read_csv(input_data_path)

# Initial inspection
with open(output_stats_path, 'w') as f:
    f.write("Initial Data Statistics:\n")
    f.write(merged_df.describe().to_string())
    f.write("\n\nMissing Values Before Cleaning:\n")
    f.write(merged_df.isnull().sum().to_string())

# Handle missing values and duplicates
cleaned_df = merged_df.fillna(0)
cleaned_df = cleaned_df.drop_duplicates()

cleaned_df.to_csv(output_data_path, index=False)

# Final inspection
with open(output_stats_path, 'a') as f:
    f.write("\n\nCleaned Data Statistics:\n")
    f.write(cleaned_df.describe().to_string())
    f.write("\n\nMissing Values After Cleaning:\n")
    f.write(cleaned_df.isnull().sum().to_string())
    f.write("\n\nNumber of duplicate rows removed: {}\n".format(len(merged_df) - len(cleaned_df)))

print("Data preprocessing complete. Cleaned data saved to:", output_data_path)
print("Statistics saved to:", output_stats_path)