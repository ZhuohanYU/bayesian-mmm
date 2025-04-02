# bayesian_mmm_cli.py

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import pytensor.tensor as pt
from pytensor.scan import scan


def load_or_generate_data(file_path=None):
    if file_path:
        data = pd.read_csv(file_path)
        data['region_code'] = pd.Categorical(data['region']).codes
    else:
        np.random.seed(42)
        n_samples = 15000
        channels = ['TV', 'Radio', 'Social', 'Print']
        regions = ['East', 'West', 'North', 'South']
        true_betas = np.array([0.05, 0.03, 0.07, 0.02])
        region_effects = np.array([-10, 5, 2, 8])

        X = np.random.normal(100, 20, size=(n_samples, len(channels)))
        region_indices = np.random.choice(len(regions), n_samples)
        y = np.dot(X, true_betas) + region_effects[region_indices] + np.random.normal(0, 2, n_samples)

        data = pd.DataFrame(X, columns=[f"{c}_spend" for c in channels])
        data['region'] = [regions[i] for i in region_indices]
        data['region_code'] = region_indices
        data['sales'] = y
    return data


def oversample_data_by_metric(df, metric_col, target_count=None, random_state=42):
    grouped = df.groupby(metric_col)
    if target_count is None:
        target_count = grouped.size().max()
    oversampled = []
    for _, group in grouped:
        if len(group) < target_count:
            oversampled.append(resample(group, replace=True, n_samples=target_count, random_state=random_state))
        else:
            oversampled.append(group)
    return pd.concat(oversampled).reset_index(drop=True)


def build_mmm_model_adstock_saturation(data, X_cols):
    X_raw = data[X_cols].astype(np.float32).values
    y = data['sales'].astype(np.float32).values
    region_codes = data['region_code'].astype(int).values
    n_regions = len(np.unique(region_codes))
    n_obs, n_channels = X_raw.shape

    with pm.Model() as model:
        X_input = pm.MutableData("X_input", X_raw)

        decay_unbounded = pm.Normal("decay_unbounded", mu=0, sigma=1, shape=n_channels)
        decay = pm.Deterministic("decay", pm.math.sigmoid(decay_unbounded))

        X_seq = X_input.T.T  # shape: (n_obs, n_channels)
        X0 = X_seq[0, :]

        def adstock_step(x_t, x_tm1, d):
            return x_t + d * x_tm1

        adstocked, _ = scan(
            fn=lambda x_t, x_tm1: adstock_step(x_t, x_tm1, decay),
            sequences=X_seq[1:],
            outputs_info=[X0],
        )

        X_adstocked = pt.concatenate([X0[None, :], adstocked], axis=0)

        alpha = pm.HalfNormal("alpha", sigma=1.0, shape=n_channels)
        X_saturated = X_adstocked / (1 + alpha * X_adstocked)

        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_channels)
        sigma_region = pm.HalfNormal("sigma_region", sigma=1.0)
        region_effects = pm.Normal("region_effects", mu=0, sigma=sigma_region, shape=n_regions)

        mu = pt.dot(X_saturated, beta) + region_effects[region_codes]
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    return model


def evaluate_model(trace, model, data):
    with model:
        posterior_predictive = pm.sample_posterior_predictive(trace, return_inferencedata=True)

    y_pred = posterior_predictive.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
    y_actual = data['sales'].values

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)

    print(f"\n✅ RMSE: {rmse:.2f}")
    print(f"✅ R²: {r2:.2f}")
    print(f"✅ MAPE: {mape:.2%}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_pred)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs. Predicted Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(posterior_predictive.posterior_predictive["y_obs"].stack(draws=("chain", "draw")).values,
                 bins=30, kde=True, color='blue', stat='density', alpha=0.5, label='Predicted')
    sns.histplot(y_actual, bins=30, kde=True, color='red', stat='density', alpha=0.5, label='Actual')
    plt.legend()
    plt.title("Posterior Predictive Check")
    plt.xlabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian MMM with adstock and saturation")
    parser.add_argument("--data", type=str, help="Path to input CSV file (optional)", default=None)
    parser.add_argument("--balance", type=str, help="Column to oversample by (e.g., 'region')", default="region")
    args = parser.parse_args()

    print("📥 Loading data...")
    data = load_or_generate_data(args.data)
    X_cols = [col for col in data.columns if "_spend" in col]

    print(f"⚖️ Oversampling by '{args.balance}'...")
    data = oversample_data_by_metric(data, metric_col=args.balance)

    scaler = StandardScaler()
    data[X_cols] = scaler.fit_transform(data[X_cols])
    data['sales'] = data['sales'].astype(np.float32)
    data['region_code'] = pd.Categorical(data['region']).codes

    print("🧠 Building model with learnable adstock and saturation...")
    model = build_mmm_model_adstock_saturation(data, X_cols)

    print("📈 Sampling...")
    with model:
        trace = pm.sample(1000, tune=1000, chains=2, cores=1,
                          target_accept=0.95, init="adapt_diag", return_inferencedata=True)

    print("📊 Evaluating model...")
    evaluate_model(trace, model, data)
    print("✅ Done.")


if __name__ == "__main__":
    main()