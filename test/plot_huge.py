import matplotlib.pyplot as plt

# Data
nodes = [100, 120, 140, 160]
methods = ['2S', 'SPO+', 'NCE']

acc = {
    '2S': [23.27, 22.77, 22.44, 23.61],
    'SPO+': [23.72, 22.99, 22.86, 24.29],
    'NCE': [20.65, 20.36, 19.98, 18.87]
}

cr = {
    '2S': [4.44, 4.34, 4.49, 4.02],
    'SPO+': [4.51, 4.36, 4.58, 4.15],
    'NCE': [4.62, 4.53, 4.63, 4.71]
}

re = {
    '2S': [4.09, 4.63, 4.98, 5.30],
    'SPO+': [4.15, 4.66, 5.18, 5.61],
    'NCE': [4.22, 5.07, 5.50, 6.33]
}

tt = {
    '2S': [42.44, 47.67, 61.54, 60.12],
    'SPO+': [303.44, 539.92, 710.41, 786.20],
    'NCE': [389.13, 663.49, 557.33, 565.57]
}

# Plotting
def plot_metric(metric_data, title, ylabel):
    for method in methods:
        plt.plot(nodes, metric_data[method], label=method, marker='o')
    plt.title(title)
    plt.xlabel('Number of Nodes')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_metric(acc, 'Accuracy vs Number of Nodes', 'Accuracy (%)')
plot_metric(cr, 'Cost Ratio vs Number of Nodes', 'Cost Ratio')
plot_metric(re, 'Regret vs Number of Nodes', 'Regret')
plot_metric(tt, 'Training Time vs Number of Nodes', 'Training Time (sec)')
