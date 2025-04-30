import matplotlib.pyplot as plt

# Data
models = ['VGAE-75', 'VGAE-50', 'VGAE-25', 'VGAE-5', 'Bilinear interpolation']
mse_values = [0.28281, 0.18007, 0.064705, 0.015372, 0.09396]

# Identify index of Bilinear interpolation for highlighting
highlight_index = models.index('Bilinear interpolation')

plt.figure(figsize=(8, 5))
colors = ['blue'] * len(models)
colors[highlight_index] = 'red'

bars = plt.bar(models, mse_values, color=colors)
plt.xlabel("Model")
plt.ylabel("Test MSE")
plt.title("Test MSE for Each Model")
plt.xticks(rotation=45, ha='right')

# Annotate each bar with its MSE value
for bar, value in zip(bars, mse_values):
    height = bar.get_height()
    plt.annotate(f'{value:.6f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # Vertical offset for the text
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.tight_layout()
plt.show()
plt.savefig('test_mse_plot.png', dpi=300)