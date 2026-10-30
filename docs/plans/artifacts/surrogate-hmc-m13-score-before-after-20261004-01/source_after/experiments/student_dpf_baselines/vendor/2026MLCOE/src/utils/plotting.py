"""
Plotting Module
===============
Standardized visualization tools for filter performance analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def setup_plotting_style():
    """Configures the aesthetic parameters for the report figures."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')

    plt.rcParams.update({
        'font.size': 11,
        'lines.linewidth': 2,
        'figure.figsize': (14, 4),
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def plot_state_tracking(true_x, estimates, labels, title="State Estimation"):
    """
    Generates a standardized state tracking plot.

    :param true_x: Ground truth trajectory.
    :param estimates: List of estimated trajectories.
    :param labels: Names of the methods.
    :param title: Plot title.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(true_x, 'k-', label='Ground Truth', alpha=0.8)

    for est, label in zip(estimates, labels):
        plt.plot(est, '--', label=label)

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.legend()
    plt.tight_layout()
