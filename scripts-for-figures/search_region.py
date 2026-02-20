# This script generates a 2D plot with a shaded region defined by a set of linear and
# non-linear constraints. It uses the matplotlib and numpy libraries.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib import rc

# Use LaTeX for text in the plot
plt.rcParams.update({"text.usetex": True})
rc('font', family='serif', weight = 'bold')


def plot_feasible_region(ax, x_range, y_range, sum_constraints, ratio_constraints, label, shade_color, zorder=None):
    """
    Plots a 2D shaded region that satisfies a given set of constraints on a
    pre-existing Axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        x_range (tuple): A tuple (x_min, x_max) for the x-axis bounds.
        y_range (tuple): A tuple (y_min, y_max) for the y-axis bounds.
        sum_constraints (tuple): A tuple (sum_min, sum_max) for x+y constraints.
        ratio_constraints (tuple): A tuple (ratio_min, ratio_max) for x/y constraints.
        label (str): The label for the shaded region in the plot legend.
        shade_color (str): The color for the shaded region (e.g., 'blue', 'green').
    """
    # Generate a grid of x and y values to check for valid points.
    x_vals = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 500)
    y_vals = np.logspace(np.log10(y_range[0]), np.log10(y_range[1]), 500)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Define the constraints as boolean masks.
    constraint1 = (X >= x_range[0]) & (X <= x_range[1])
    constraint2 = (Y >= y_range[0]) & (Y <= y_range[1])
    
    constraint3 = (X + Y >= sum_constraints[0]) & (X + Y <= sum_constraints[1])

    # Handle the x/y constraint, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        constraint4 = (X / Y >= ratio_constraints[0]) & (X / Y <= ratio_constraints[1])

    # Combine all the constraints to find the final feasible region.
    feasible_region = constraint1 & constraint2 & constraint3 & constraint4

    # Create a custom colormap for the shaded region
    cmap = ListedColormap(['none', shade_color])

    # Create a masked array for the feasible region and plot it.
    ax.imshow(feasible_region, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
              origin='lower', cmap=cmap, alpha=1, zorder=zorder)
    
    # Plot a dummy line to create a legend entry for the shaded region
    ax.plot([], [], color=shade_color, alpha=1, linewidth=10, label=label)
    
# Example usage of the function
if __name__ == "__main__":
    # Create the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set log scales for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set axis labels with LaTeX formatting and specific font sizes
    ax.set_xlabel('${m_1^{\mathsf{det}}} (M_{\odot})$', fontsize=24)
    ax.set_ylabel('$m_2^{\mathsf{det}} (M_{\odot})$', fontsize=24)
    
    # Set the title and grid
    #ax.set_title("Feasible Regions for Multiple Constraints")
    ax.grid(True)
    
    # Set fontsize for ticks
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    # Our search
    x_bounds1 = (15, 70)
    y_bounds1 = (3, 10)
    sum_bounds1 = (18, 100)
    ratio_bounds1 = (5, 12) 
    plot_feasible_region(ax, x_bounds1, y_bounds1, sum_bounds1, ratio_bounds1, "This work and Schmidt et al. (2024)", "darkorange")

    # Charlies search
    x_bounds2 = (5, 20)
    y_bounds2 = (1.2, 1.7)
    sum_bounds2 = (6.2, 21.7)
    ratio_bounds2 = (2.94, 11.76)
    plot_feasible_region(ax, x_bounds2, y_bounds2, sum_bounds2, ratio_bounds2, "Harry and Hoy (2024), precessing NSBH search", "limegreen", zorder=5)

    # S4 search
    #x_bounds3 = (1, 30)
    #y_bounds3 = (1.0, 3)
    #sum_bounds3 = (2, 33.0)
    #ratio_bounds3 = (1, 30.0)
    #plot_feasible_region(ax, x_bounds3, y_bounds3, sum_bounds3, ratio_bounds3, "Fairhurst and Harry (2011)", "lightblue")
    
    # Add a single legend for all plots
    ax.legend(loc='upper left', prop={'size': 15})
   
    plt.xlim([4.0, 100])
    plt.ylim([1.0, 15]) 
    plt.tight_layout()
    plt.savefig('Paper-plots/past-precessing-searches.png', dpi=600)
    plt.show()
