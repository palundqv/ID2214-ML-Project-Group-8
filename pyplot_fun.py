import matplotlib.pyplot as plt

# Data
num_features = [10, 30, 50, 80, 100, 722]
rf_with_smote = [0.7211, 0.7650, 0.7663, 0.7537, 0.7643, 0.7933]
rf_without_smote = [0.7323, 0.7595, 0.7570, 0.7572, 0.7676, 0.7760]
mlp_with_smote = [0.6377, 0.6592, 0.6304, 0.5185, 0.5193, 0.5172]
mlp_without_smote = [0.6700, 0.6784, 0.6838, 0.5154, 0.5251, 0.5083]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(num_features, rf_with_smote, marker='o', label='RF with SMOTE')
plt.plot(num_features, rf_without_smote, marker='o', label='RF without SMOTE')
plt.plot(num_features, mlp_with_smote, marker='o', label='MLP with SMOTE')
plt.plot(num_features, mlp_without_smote, marker='o', label='MLP without SMOTE')

plt.xscale('log')
plt.xticks(num_features, num_features)
plt.title('Model Performance Comparison (Logarithmic Scale)')
plt.xlabel('Number of Features')
plt.ylabel('AUC score')
plt.legend()
plt.grid(True)
plt.show()
