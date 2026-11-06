"""Draw assets/training_curves.png from the two training logs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent


def load(name):
    return list(csv.DictReader((root / "results" / name).open()))


a, b = load("training_cnn_lstm.csv"), load("training_lstm_hourly.csv")
BG, INK, MUT, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=200, facecolor=BG)
for ax in (a1, a2):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT)
    ax.set_xlabel("Epoch", color=MUT)
    ax.set_ylabel("MSE loss (normalized)", color=MUT)

a1.plot([int(r["epoch"]) for r in a], [float(r["loss"]) for r in a], color=BLUE, lw=2, marker="o", label="Train")
a1.plot([int(r["epoch"]) for r in a], [float(r["val_loss"]) for r in a], color=ORANGE, lw=2, marker="o", label="Validation")
a1.set_title("CNN-LSTM (daily California maps): both fall", loc="left", fontsize=11, fontweight="bold", color=INK)
a1.set_xticks(range(1, 6))
a1.legend(frameon=False)

a2.plot([int(r["epoch"]) for r in b], [float(r["loss"]) for r in b], color=BLUE, lw=2, label="Train")
a2.plot([int(r["epoch"]) for r in b], [float(r["val_loss"]) for r in b], color=ORANGE, lw=2, label="Validation")
a2.set_title("LSTM (hourly cells): validation drifts up", loc="left", fontsize=11, fontweight="bold", color=INK)
a2.legend(frameon=False)

fig.tight_layout()
out = root / "assets" / "training_curves.png"
fig.savefig(out, facecolor=BG)
print("wrote", out)
