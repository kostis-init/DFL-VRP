import pandas as pd
import matplotlib.pyplot as plt

# Hard-coded data
data40 = pd.DataFrame({
    'Solving (%)': [5, 5, 10, 10, 25, 25, 50, 50, 100, 100],
    'Method': ['SP_LR', 'NC_LR', 'SP_LR', 'NC_LR', 'SP_LR', 'NC_LR', 'SP_LR', 'NC_LR', 'SP_LR', 'NC_LR'],
    'TT (sec)': [441.09, 220.97, 387.04, 475.79, 500.93, 513.38, 641.40, 1360.71, 1028.16, 2365.32],
    'Nodes': [40]*10
})

data80 = pd.DataFrame({
    'Solving (%)': [5, 5, 10, 10, 25, 25, 50, 50, 100, 100],
    'Method': ['SP_DL', 'NC_DL', 'SP_DL', 'NC_DL', 'SP_DL', 'NC_DL', 'SP_DL', 'NC_DL', 'SP_DL', 'NC_DL'],
    'TT (sec)': [774.37, 498.89, 665.21, 585.44, 5873.70, 888.94, 4685.73, 3077.54, 8358.05, 6294.36],
    'Nodes': [80]*10
})

# Combine the data
data = pd.concat([data40, data80])

# Create a figure
plt.figure(figsize=(10, 6))
plt.title('TT (sec) vs Solving Percentage')
plt.xlabel('Solving Percentage (%)')
plt.ylabel('TT (sec)')

# Plot each method
for method in ['SP_LR', 'NC_LR', 'SP_DL', 'NC_DL']:
    subset = data[data['Method'] == method]
    plt.plot(subset['Solving (%)'], subset['TT (sec)'], marker='o', label=f'{method}')

plt.legend()
plt.grid(True)
plt.savefig("caching_comp.png", dpi=300)

plt.show()
