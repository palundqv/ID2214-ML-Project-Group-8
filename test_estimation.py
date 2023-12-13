import pandas as pd
import numpy as np

import pandas as pd

def write_predictions_to_txt(auc_estimate, predicted_probabilities, output_file_path='output.txt'):
    # Create a DataFrame with the AUC estimate and predicted probabilities
    data = {'Probability': [auc_estimate] + predicted_probabilities}
    df = pd.DataFrame(data)

    # Write the DataFrame to a text file
    df.to_csv(output_file_path, header=False, index=False, sep='\t')

# Example usage:
# Replace the following with your actual AUC estimate and predicted probabilities
auc_estimate = 0.85
predicted_probabilities = [0.6, 0.7, 0.8, ...]  # Replace with your actual values

write_predictions_to_txt(auc_estimate, predicted_probabilities)

predictions_df = pd.read_csv("8.txt", header=None)
assert predictions_df.shape == (51077, 1)
assert np.all((predictions_df.values >= 0) & (predictions_df.values <= 1))