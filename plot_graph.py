import matplotlib.pyplot as plt
import numpy as np

# Mask ratios representing the models (from both the Full Loss Table and Masked Loss table)
mask_ratios = ["5", "25", "50", "75", "100"]

# Full Loss Table values (rounded to 4-digit accuracy)
full_loss_test_mse         = [0.0035, 0.0149, 0.0301, 0.0446, 0.0606]
full_loss_test_mae         = [0.0169, 0.0469, 0.0837, 0.1202, 0.1579]
full_loss_test_mse_masked  = [0.0669, 0.0588, 0.0599, 0.0593, 0.0606]
full_loss_test_mae_masked  = [0.1677, 0.1551, 0.1558, 0.1552, 0.1579]

# Masked Loss table values (rounded to 4-digit accuracy)
masked_loss_test_mse       = [0.0772, 0.0692, 0.0660, 0.0629, 0.0606]
masked_loss_test_mae       = [0.1818, 0.1689, 0.1660, 0.1611, 0.1579]
masked_loss_test_mse_masked= [0.0739, 0.0648, 0.0630, 0.0612, 0.0606]
masked_loss_test_mae_masked= [0.1782, 0.1635, 0.1617, 0.1589, 0.1579]

# For masked plots additional bilinear bar values:
masked_bilinear_mse = [0.09110, 0.0963, 0.0938, 0.0937, 0.0939]
masked_bilinear_mae = [0.1891, 0.1924, 0.1902, 0.1903, 0.1905]

# Function to plot two bars with a horizontal red dashed bilinear line.
def plot_grouped_bar_with_line(mask_ratios, data_full, data_masked, bilinear_val, metric_name):
    n = len(mask_ratios)
    x = np.arange(n)      # x-axis positions for each category
    width = 0.35          # width for each bar

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the bar groups
    ax.bar(x - width/2, data_full, width, color='C0', label='Full Loss')
    ax.bar(x + width/2, data_masked, width, color='C1', label='Masked Loss')
    
    # Draw horizontal red dashed line for the bilinear threshold.
    # Set zorder low so that the numbers (drawn later) appear on top.
    ax.axhline(bilinear_val, color='red', linestyle='--', label='Bilinear', zorder=0)

    # Compute an offset based on the highest value among the bars and bilinear value
    max_val = max(max(data_full), max(data_masked), bilinear_val)
    offset = 0.05 * max_val if max_val != 0 else 0.005

    # Set y-limit to provide extra space above the highest element
    ax.set_ylim(0, max_val + 2 * offset)

    # Annotate the bars using the computed offset so there is a consistent gap.
    for i in range(n):
        ax.text(x[i] - width/2, data_full[i] + offset, f'{data_full[i]:.4f}', 
                ha='center', va='bottom', fontsize=9)
        ax.text(x[i] + width/2, data_masked[i] + offset, f'{data_masked[i]:.4f}', 
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(mask_ratios)
    ax.set_xlabel("Masking Ratio (%)")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)

    # Put legend above the plot.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Function to plot three bars (including an extra red bar for bilinear) for the masked metrics.
def plot_grouped_bar_with_extra_bar(mask_ratios, data_full, data_masked, data_bilinear, metric_name):
    n = len(mask_ratios)
    x = np.arange(n)      # positions on the x-axis for each category
    width = 0.25          # a slightly narrower bar width to accommodate three bars

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot three sets of bars: Full Loss, Masked Loss, and the extra Bilinear.
    ax.bar(x - width, data_full, width, color='C0', label='Full Loss')
    ax.bar(x, data_masked, width, color='C1', label='Masked Loss')
    ax.bar(x + width, data_bilinear, width, color='red', label='Bilinear')

    # Compute an offset based on the maximum value among all three data sets.
    max_val = max(max(data_full), max(data_masked), max(data_bilinear))
    offset = 0.05 * max_val if max_val != 0 else 0.005

    # Set y-axis limit for a comfortable top margin.
    ax.set_ylim(0, max_val + 2 * offset)

    # Annotate each bar with its corresponding value using the computed offset.
    for i in range(n):
        ax.text(x[i] - width, data_full[i] + offset, f'{data_full[i]:.4f}',
                ha='center', va='bottom', fontsize=9)
        ax.text(x[i], data_masked[i] + offset, f'{data_masked[i]:.4f}',
                ha='center', va='bottom', fontsize=9)
        ax.text(x[i] + width, data_bilinear[i] + offset, f'{data_bilinear[i]:.4f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(mask_ratios)
    ax.set_xlabel("Masking Ratio (%)")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)

    # Place legend above the plot.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{metric_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate the four plots with improved spacing:

# 1. Test MSE with horizontal bilinear line (value 0.0939)
plot_grouped_bar_with_line(mask_ratios, full_loss_test_mse, masked_loss_test_mse, 0.0939, 'Test MSE')

# 2. Test MAE with horizontal bilinear line (value 0.1905)
plot_grouped_bar_with_line(mask_ratios, full_loss_test_mae, masked_loss_test_mae, 0.1905, 'Test MAE')

# 3. Test MSE masked with an extra bilinear bar
plot_grouped_bar_with_extra_bar(mask_ratios, full_loss_test_mse_masked, masked_loss_test_mse_masked, masked_bilinear_mse, 'Test MSE masked')

# 4. Test MAE masked with an extra bilinear bar
plot_grouped_bar_with_extra_bar(mask_ratios, full_loss_test_mae_masked, masked_loss_test_mae_masked, masked_bilinear_mae, 'Test MAE masked')
