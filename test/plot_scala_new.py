import matplotlib.pyplot as plt

# Data from the table
nodes = [10, 20, 40, 60, 80]
methods = ['2S', 'SPO+', 'NCE']

# Accuracy (ACC)
acc = {
    '2S': [32.36, 28.92, 16.49, 6.99, 6.52],
    'SPO+': [36.31, 31.48, 21.10, 17.05, 16.42],
    'NCE': [34.22, 29.95, 23.03, 15.82, 14.91]
}

# Cost Ratio (CR)
cr = {
    '2S': [2.73, 5.03, 8.22, 8.68, 8.81],
    'SPO+': [3.41, 4.56, 6.25, 6.01, 6.14],
    'NCE': [3.39, 5.19, 6.81, 6.76, 6.42]
}

# Regret (RE)
re = {
    '2S': [0.71, 1.32, 3.29, 5.16, 6.21],
    'SPO+': [0.99, 1.10, 2.53, 3.16, 3.95],
    'NCE': [0.98, 1.29, 2.60, 3.79, 4.11]
}

# Training Time (TT)
tt = {
    '2S': [14.88, 26.01, 78.69, 483.03, 1874.56],
    'SPO+': [189.82, 293.77, 619.23, 1349.57, 2153.86],
    'NCE': [227.26, 269.17, 545.12, 1135.43, 2001.51]
}

# Plotting function
def plot_data(data, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(nodes, data[method], label=method, marker='o')
    plt.xlabel('Number of Nodes')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Generate the plots
plot_data(acc, 'Accuracy (%)', 'Accuracy vs. Number of Nodes', 'accuracy_plot.png')
plot_data(cr, 'Cost Ratio', 'Cost Ratio vs. Number of Nodes', 'cost_ratio_plot.png')
plot_data(re, 'Regret', 'Regret vs. Number of Nodes', 'regret_plot.png')
plot_data(tt, 'Training Time (sec)', 'Training Time vs. Number of Nodes', 'training_time_plot.png')
