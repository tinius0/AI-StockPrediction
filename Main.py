# predictor.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from stock_model import prepare_data


def rolling_mae(y_true: np.ndarray, y_pred: np.ndarray, window: int = 20) -> np.ndarray:
    errors = np.abs(y_true - y_pred)
    return np.array([errors[max(0, i - window + 1): i + 1].mean() for i in range(len(errors))])


def main(ticker: str = "AAPL", save_path: str = "predictor_output.png"):
    X, y, _ = prepare_data(ticker)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmae = rolling_mae(y_test, preds)

    print(f"\n--- {ticker} Predictor ---")
    print(f"Overall MAE         : ${mae:.2f}")
    print(f"Final rolling MAE   : ${rmae[-1]:.2f}\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")

    #Price prediction
    ax1.set_facecolor("#0f0f0f")
    ax1.plot(y_test, label="Actual",    color="#ffffff", alpha=0.7, linewidth=1.2)
    ax1.plot(preds,  label="Predicted", color="#ff6b35", linestyle="--", linewidth=1.5)
    ax1.set_title(f"{ticker} — Next-Day Close Prediction", color="white", fontsize=13, pad=10)
    ax1.set_ylabel("Price (USD)", color="#aaaaaa")
    ax1.legend(framealpha=0.2, labelcolor="white")
    ax1.tick_params(colors="#aaaaaa")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    #Rolling MAE
    ax2.set_facecolor("#0f0f0f")
    ax2.plot(rmae, color="#f0c040", linewidth=1.5)
    ax2.axhline(mae, color="#888888", linestyle=":", linewidth=1, label=f"Overall MAE ${mae:.2f}")
    ax2.set_title("Rolling 20-Day MAE (Model Drift)", color="white", fontsize=12, pad=10)
    ax2.set_ylabel("MAE (USD)", color="#aaaaaa")
    ax2.set_xlabel("Test Day", color="#aaaaaa")
    ax2.legend(framealpha=0.2, labelcolor="white")
    ax2.tick_params(colors="#aaaaaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"Saved -> {save_path}")
    plt.show()


if __name__ == "__main__":
    main("AAPL")