import matplotlib.pyplot as plt

# Data from the tables
solving_percentages = [5, 10, 25, 50, 100]
methods = ['SPO+', 'NCE']

# Training Time (TT) for 40 nodes
tt_40 = {
    'SPO+': [441.09, 387.04, 500.93, 641.40, 1028.16],
    'NCE': [220.97, 475.79, 513.38, 1360.71, 2365.32]
}

# Regret (RE) for 40 nodes
re_40 = {
    'SPO+': [2.16, 2.05, 1.93, 2.01, 1.97],
    'NCE': [2.27, 2.40, 2.20, 2.40, 2.29]
}

# Training Time (TT) for 80 nodes
tt_80 = {
    'SPO+': [774.37, 665.21, 5873.70, 4685.73, 8358.05],
    'NCE': [498.89, 585.44, 888.94, 3077.54, 6294.36]
}

# Regret (RE) for 80 nodes
re_80 = {
    'SPO+': [3.43, 3.42, 3.48, 3.21, 3.29],
    'NCE': [3.51, 3.45, 3.63, 3.69, 3.64]
}

# Plotting function
def plot_data(data_40, data_80, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(solving_percentages, data_40[method], label=f"{method} (40 nodes)", marker='o')
        plt.plot(solving_percentages, data_80[method], label=f"{method} (80 nodes)", marker='o', linestyle='--')
    plt.xlabel('Solving Percentage (%)')
    plt.xticks(solving_percentages)  # Set x-axis ticks to only the solving percentages
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Generate the plots
plot_data(tt_40, tt_80, 'Training Time (sec)', 'Training Time vs. Solving Percentage', 'cach_training_time_plot.png')
plot_data(re_40, re_80, 'Regret', 'Regret vs. Solving Percentage', 'cach_regret_plot.png')
