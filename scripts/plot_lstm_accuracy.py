from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "lstm" / "walk_forward_metrics.csv"
OUTPUT = ROOT / "lstm" / "figures" / "lstm_accuracy_line.png"


def main() -> None:
    metrics = pd.read_csv(INPUT)
    test = metrics[(metrics["model"] == "lstm") & (metrics["split"] == "test")].copy()
    test = test.sort_values("fold")

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.plot(
        test["fold"],
        test["accuracy"] * 100,
        color="#2563EB",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="LSTM 测试准确率",
    )
    axis.plot(
        test["fold"],
        test["majority_baseline_accuracy"] * 100,
        color="#94A3B8",
        marker="o",
        linewidth=2,
        linestyle="--",
        label="多数类基准准确率",
    )

    for _, row in test.iterrows():
        axis.annotate(
            f"{row['accuracy'] * 100:.2f}%",
            (row["fold"], row["accuracy"] * 100),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#1D4ED8",
        )

    axis.set_title("LSTM Walk-forward 测试准确率", fontsize=16, pad=14)
    axis.set_xlabel("测试折数")
    axis.set_ylabel("准确率（%）")
    axis.set_xticks(test["fold"])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
