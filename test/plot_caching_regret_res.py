import pandas as pd
import matplotlib.pyplot as plt

# Hard-coded data
data40 = pd.DataFrame({
    'Solving (%)': [5, 5, 10, 10, 25, 25, 50, 50, 100, 100],
    'Method': ['SP_LR', 'NC_LR', 'SP_LR', 'NC_LR', 'SP_LR', 'NC_LR', 'SP_LR', 'NC_LR', 'SP_LR', 'NC_LR'],
    'RE': [2.16, 2.27, 2.05, 2.40, 1.93, 2.20, 2.01, 2.40, 1.97, 2.29],
    'Nodes': [40]*10
})

data80 = pd.DataFrame({
    'Solving (%)': [5, 5, 10, 10, 25, 25, 50, 50, 100, 100],
    'Method': ['SP_DL', 'NC_DL', 'SP_DL', 'NC_DL', 'SP_DL', 'NC_DL', 'SP_DL', 'NC_DL', 'SP_DL', 'NC_DL'],
    'RE': [3.43, 3.51, 3.42, 3.45, 3.48, 3.63, 3.21, 3.69, 3.29, 3.64],
    'Nodes': [80]*10
})

# Combine the data
data = pd.concat([data40, data80])

# Create a figure
plt.figure(figsize=(10, 6))
plt.title('RE vs Solving Percentage')
plt.xlabel('Solving Percentage (%)')
plt.ylabel('RE')

# Plot each method
for method in ['SP_LR', 'NC_LR', 'SP_DL', 'NC_DL']:
    subset = data[data['Method'] == method]
    plt.plot(subset['Solving (%)'], subset['RE'], marker='o', label=f'{method}')

plt.legend()
plt.grid(True)
plt.savefig("caching_comp_regret.png", dpi=300)

plt.show()
