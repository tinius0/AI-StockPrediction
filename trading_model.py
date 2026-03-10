# trading_system.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from stock_model import prepare_data, get_feature_names


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    mean_r = returns.mean()
    std_r  = returns.std()
    return (mean_r / std_r * np.sqrt(periods_per_year)) if std_r != 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def run_trading_system(ticker: str = "AAPL", initial_capital: float = 10_000.0):
    X, y, _ = prepare_data(ticker)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    current_prices = X_test[:, 0]
    signals = np.sign(preds - current_prices)
    actual_returns = (y_test - current_prices) / current_prices
    strategy_returns = signals * actual_returns

    equity_curve    = initial_capital * np.cumprod(1 + strategy_returns)
    benchmark_curve = initial_capital * np.cumprod(1 + actual_returns)

    mae       = mean_absolute_error(y_test, preds)
    final_ret = (equity_curve[-1] / initial_capital - 1) * 100
    bench_ret = (benchmark_curve[-1] / initial_capital - 1) * 100
    sr        = sharpe_ratio(strategy_returns)
    mdd       = max_drawdown(equity_curve) * 100
    win_rate  = (strategy_returns > 0).mean() * 100

    print(f"\n{'='*42}")
    print(f"  {ticker} Trading System Results")
    print(f"{'='*42}")
    print(f"  MAE (price prediction)  : ${mae:.2f}")
    print(f"  Strategy total return   : {final_ret:+.2f}%")
    print(f"  Benchmark total return  : {bench_ret:+.2f}%")
    print(f"  Sharpe ratio            : {sr:.2f}")
    print(f"  Max drawdown            : {mdd:.2f}%")
    print(f"  Win rate                : {win_rate:.1f}%")
    print(f"{'='*42}\n")

    return equity_curve, benchmark_curve, preds, y_test, model


def visualize_performance(equity, benchmark, preds, y_test, model, ticker,
                          save_path: str = "trading_system_output.png"):
    feature_names = get_feature_names()
    importances   = model.feature_importances_
    order         = np.argsort(importances)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f0f0f")

    #Equity curves
    ax1 = axes[0]
    ax1.set_facecolor("#0f0f0f")
    ax1.plot(equity,    label="AI Strategy", color="#00ff88", linewidth=2)
    ax1.plot(benchmark, label="Buy & Hold",  color="#888888", linewidth=1.5, linestyle="--")
    ax1.set_title(f"Portfolio Growth: {ticker}", color="white", fontsize=13, pad=12)
    ax1.set_ylabel("Value (USD)", color="#aaaaaa")
    ax1.set_xlabel("Trading Days (Test Set)", color="#aaaaaa")
    ax1.tick_params(colors="#aaaaaa")
    ax1.legend(framealpha=0.2, labelcolor="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    #Feature importances
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f0f")
    ax2.barh([feature_names[i] for i in order], importances[order],
              color="#00aaff", alpha=0.85)
    ax2.set_title("Feature Importances", color="white", fontsize=13, pad=12)
    ax2.set_xlabel("Importance", color="#aaaaaa")
    ax2.tick_params(colors="#aaaaaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"Saved -> {save_path}")
    plt.show()


if __name__ == "__main__":
    results = run_trading_system("AAPL")
    visualize_performance(*results, "AAPL")