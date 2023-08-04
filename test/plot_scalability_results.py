import matplotlib.pyplot as plt

# Define the node sizes
node_sizes = [10, 20, 40, 80, 120, 160]


data = {
    "2S_LR": {
        "Accuracy (%)": [20.71, 15.04, 14.55, 13.92, 13.70, 12.91],
        "Cost Ratio": [4.19, 4.63, 5.17, 5.51, 6.87, 6.26],
        "Regret": [1.17, 2.10, 3.24, 5.06, 7.83, 9.83],
        "Train Time (sec)": [17.82, 36.01, 103.69, 630.56, 1947.68, 7034.68]
    },
    "2S_DL": {
        "Accuracy (%)": [44.00, 31.11, 25.55, 23.75, 24.07, 25.42],
        "Cost Ratio": [2.64, 3.24, 3.57, 3.95, 4.34, 4.04],
        "Regret": [0.60, 1.30, 2.00, 3.30, 4.46, 5.68],
        "Train Time (sec)": [9.92, 15.30, 126.77, 196.00, 187.79, 487.39]
    },
    "SP_LR": {
        "Accuracy (%)": [46.54, 29.74, 26.52, None, None, None],
        "Cost Ratio": [2.29, 3.70, 3.63, None, None, None],
        "Regret": [0.48, 1.57, 2.05, None, None, None],
        "Train Time (sec)": [129.82, 362.42, 387.03, None, None, None]
    },
    "SP_DL": {
        "Accuracy (%)": [44.74, 30.48, 26.40, 22.92, 24.66, 23.25],
        "Cost Ratio": [2.50, 3.28, 3.69, 4.05, 4.45, 4.32],
        "Regret": [0.55, 1.32, 2.10, 3.42, 4.61, 6.20],
        "Train Time (sec)": [116.62, 273.77, 301.67, 665.21, 1407.17, 3335.99]
    },
    "NC_LR": {
        "Accuracy (%)": [46.06, 30.14, 23.93, None, None, None],
        "Cost Ratio": [2.63, 3.68, 4.08, None, None, None],
        "Regret": [0.60, 1.56, 2.40, None, None, None],
        "Train Time (sec)": [63.58, 249.17, 475.79, None, None, None]
    },
    "NC_DL": {
        "Accuracy (%)": [44.44, 28.79, 22.20, 21.94, 23.90, 21.94],
        "Cost Ratio": [2.48, 3.70, 4.18, 4.07, 4.55, 4.40],
        "Regret": [0.55, 1.57, 2.47, 3.45, 4.74, 6.34],
        "Train Time (sec)": [67.00, 242.48, 339.06, 585.44, 1091.95, 1153.85]
    }
}

# Create a separate graph for each metric
for metric in ["Accuracy (%)", "Cost Ratio", "Regret", "Train Time (sec)"]:
    plt.figure(figsize=(10, 6))
    plt.title(metric + " vs VRP Size")
    plt.xlabel("VRP Size (# Nodes)")
    plt.ylabel(metric)

    # Plot a line for each method
    for method, metrics in data.items():
        plt.plot(node_sizes, metrics[metric], label=method)

    plt.xticks(node_sizes)  # Only show the node sizes on the x-axis
    plt.ylim(bottom=0)  # Start the y-axis at 0
    plt.legend()
    plt.grid(True)

    # Save the figure to a PNG file
    plt.savefig(metric + "_vs_Node_Size.png", dpi=300)

    plt.show()
