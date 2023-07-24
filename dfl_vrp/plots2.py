import matplotlib.pyplot as plt
import numpy as np

# Given data
models = {
    "two_stage": {
        "mean": (0.20799142226458348, 5.830229653076833, 166103.57041717318, 938.7495269775391, 275.89084911346436),
        "std": (0.01, 0.5, 10000, 50, 10)
    },
    "spo": {
        "mean": (0.2568178772377863, 4.925915292343946, 135005.702430385, 9937.693444013596, 167.28093600273132),
        "std": (0.01, 0.5, 10000, 50, 10)
    },
    "spoNo": {
        "mean": (0.23849214866865462, 5.198049519994431, 144363.94625977965, 9707.711451292038, 142.9716169834137),
        "std": (0.01, 0.5, 10000, 50, 10)
    },
    "nce": {
        "mean": (0.23995299752072372, 5.178173448498041, 143680.44119303505, 9252.165268182755, 155.78800296783447),
        "std": (0.01, 0.5, 10000, 50, 10)
    },
    "nceNo": {
        "mean": (0.23967658488390758, 5.167563945025372, 143315.5980005291, 10660.7042760849, 155.2151279449463),
        "std": (0.01, 0.5, 10000, 50, 10)
    }
}

measures = ["Accuracy", "Cost Ratio", "Mean Regret", "Train Time", "Test Time"]

# X-axis labels
model_names = list(models.keys())

# Colors for the bars
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Add more colors if needed

# For each measure
for i, measure in enumerate(measures):
    plt.figure(figsize=(10, 6))

    # Values for the current measure for each model
    mean_values = [models[model_name]["mean"][i] for model_name in model_names]
    std_values = [models[model_name]["std"][i] for model_name in model_names]

    # Create bar plot with error bars and colors
    plt.bar(model_names, mean_values, yerr=std_values, color=colors[:len(model_names)])

    # Set title and labels
    plt.title(measure)
    plt.xlabel('Model')
    plt.ylabel('Value')

    # Display plot
    plt.show()
