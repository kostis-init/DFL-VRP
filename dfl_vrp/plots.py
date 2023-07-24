import matplotlib.pyplot as plt
import numpy as np

# Given data
models = {
    "two_stage": (0.20799142226458348, 5.830229653076833, 166103.57041717318, 938.7495269775391, 275.89084911346436),
    "spo": (0.2568178772377863, 4.925915292343946, 135005.702430385, 9937.693444013596, 167.28093600273132),
    "nce": (0.23995299752072372, 5.178173448498041, 143680.44119303505, 9252.165268182755, 155.78800296783447),
    "spo_no_true_costs": (0.23849214866865462, 5.198049519994431, 144363.94625977965, 9707.711451292038, 142.9716169834137),
    "nce_no_true_costs": (
    0.23967658488390758, 5.167563945025372, 143315.5980005291, 10660.7042760849, 155.2151279449463)
}

# Measure names (Replace with actual measure names)
measures = ["Accuracy", "Cost Ratio", "Mean Regret", "Train Time", "Test Time"]

# X-axis labels
model_names = list(models.keys())

# For each measure
for i, measure in enumerate(measures):
    plt.figure(figsize=(10, 6))

    # Values for the current measure for each model
    values = [models[model_name][i] for model_name in model_names]

    # Create bar plot
    plt.bar(model_names, values)

    # Set title and labels
    plt.title(measure)
    plt.xlabel('Model')
    plt.ylabel('Value')

    # Display plot
    plt.show()
