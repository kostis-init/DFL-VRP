import matplotlib.pyplot as plt
import numpy as np

# Data
solvers = ['ALNS 0.5 sec', 'ALNS 2 sec', 'Gurobi 0.5 sec', 'Gurobi 50% MIP gap', 'Gurobi 20% MIP gap', 'Exact Solving']
methods = ['SPO+', 'NCE']

# ACC values
acc_spo = [32.31, 29.89, 30.04, 30.22, 31.67, 32.77]
acc_nce = [30.51, 30.13, 30.15, 28.75, 30.62, 30.76]

# CR values
cr_spo = [4.51, 4.56, 4.62, 4.74, 4.54, 4.25]
cr_nce = [5.01, 4.67, 5.19, 5.02, 4.97, 4.99]

# RE values
re_spo = [1.02, 1.11, 1.14, 1.09, 1.06, 1.01]
re_nce = [1.19, 1.14, 1.21, 1.21, 1.24, 1.10]

# Training Time values
tt_spo = [272.12, 491.52, 300.18, 469.63, 711.97, 1126.89]
tt_nce = [238.15, 501.52, 312.54, 492.86, 692.61, 1086.53]

bar_width = 0.35
index = np.arange(len(solvers))

def plot_grouped_barchart(data_spo, data_nce, ylabel, title, filename, ylim=None):
    fig, ax = plt.subplots()
    bar1 = ax.bar(index, data_spo, bar_width, label='SPO+', color='#1f77b4', alpha=0.7)
    bar2 = ax.bar(index + bar_width, data_nce, bar_width, label='NCE', color='#ff7f0e', alpha=0.7)

    # Labeling, Title and Axes Configuration
    ax.set_xlabel('Solver')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(solvers, rotation=45)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plotting
plot_grouped_barchart(acc_spo, acc_nce, 'Accuracy (%)', 'Accuracy by Solver and Method', 'acc_solver.png', ylim=[25, 35])
plot_grouped_barchart(cr_spo, cr_nce, 'Cost Ratio', 'Cost Ratio by Solver and Method', 'cr_solver.png', ylim=[4, 5.5])
plot_grouped_barchart(re_spo, re_nce, 'Regret', 'Regret by Solver and Method', 're_solver.png', ylim=[0.8, 1.3])
plot_grouped_barchart(tt_spo, tt_nce, 'Training Time (sec)', 'Training Time by Solver and Method', 'tt_solver.png')
