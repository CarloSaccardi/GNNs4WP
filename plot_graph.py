import matplotlib.pyplot as plt
import numpy as np

# ── data ────────────────────────────────────────────────────────────────
mask_ratios = ["5", "25", "50", "75", "100"]
full_loss_test_mse = [0.0035, 0.0149, 0.0301, 0.0446, 0.0620]

x = np.arange(len(mask_ratios))

# colour palette (last two bars in orange / red)
colors = ['C0'] * (len(mask_ratios))

# ── plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(x, full_loss_test_mse, width=0.6, color=colors)

# dynamic offset: 3 % of the tallest bar
offset = 0.03 * max(full_loss_test_mse)

# annotate each bar
for bar, val in zip(bars, full_loss_test_mse):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + offset,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9)

# y-limit so annotations always fit
ax.set_ylim(0, max(full_loss_test_mse) * 1.15)

# cosmetics
ax.set_xticks(x)
ax.set_xticklabels(mask_ratios, rotation=20, ha='right')
ax.set_xlabel("Masking Ratio")
ax.set_ylabel("Test MSE")
ax.set_title("Full-loss Test MSE")
ax.grid(axis='y', linestyle='--', alpha=0.3)

fig.tight_layout()
plt.savefig("FullLoss_TestMSE.png", dpi=300, bbox_inches="tight")
plt.show()
