import matplotlib.pyplot as plt
import numpy as np

def plot_roc_auc(models_results):
    """
    Compare different ML models based on ROC AUC scores and plot the results.

    Parameters:
    - models_results (dict): A dictionary containing model names as keys and their corresponding ROC AUC scores as values.
    """

    # Sort models based on ROC AUC scores
    sorted_models = sorted(models_results.items(), key=lambda x: x[1], reverse=True)
    
    # Extract model names and ROC AUC scores for plotting
    model_names, roc_auc_scores = zip(*sorted_models)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(model_names))
    
    ax.barh(y_pos, roc_auc_scores, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()  # Invert y-axis for better visualization
    ax.set_xlabel('ROC AUC Score')
    ax.set_title('Comparison of ML Models based on ROC AUC Score')
    
    # Display the ROC AUC scores on each bar
    for i, score in enumerate(roc_auc_scores):
        ax.text(score + 0.005, i, f'{score:.4f}', ha='left', va='center')

    plt.show()



if __name__ == "__main__":
    # Example usage:
    models_results = {
    'Model1': 0.85,
    'Model2': 0.78,
    'Model3': 0.92,
    'Model4': 0.88,
}

    plot_roc_auc(models_results)
