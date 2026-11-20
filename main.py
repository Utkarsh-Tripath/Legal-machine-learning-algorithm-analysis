from pathlib import Path
import os
import warnings

# Enable local file store tracking backend in MLflow 3+
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Suppress KMeans memory leak warning on Windows with MKL
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer

from hmmlearn.hmm import GaussianHMM


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

DATA_FILE = Path("10_Property_stolen_and_recovered.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Indian_Property_Crime_ML")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. Put the CSV beside main.py."
    )


# --------------------------------------------------
# 2. LOAD AND INSPECT DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

required_columns = [
    "Area_Name",
    "Year",
    "Group_Name",
    "Cases_Property_Recovered",
    "Cases_Property_Stolen",
    "Value_of_Property_Recovered",
    "Value_of_Property_Stolen",
]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nYear range:", df["Year"].min(), "-", df["Year"].max())
print("\nMissing values:\n", df.isnull().sum())

# The Total Property rows summarize property crime categories.
# Use these rows for state/year-level analysis to avoid
# adding individual categories on top of their totals.
total_df = df[
    df["Group_Name"].astype(str).str.strip().eq("Total Property")
].copy()

if total_df.empty:
    raise ValueError("No 'Total Property' rows were found.")

numeric_features = [
    "Cases_Property_Stolen",
    "Cases_Property_Recovered",
    "Value_of_Property_Stolen",
    "Value_of_Property_Recovered",
]

total_df[numeric_features] = total_df[numeric_features].apply(
    pd.to_numeric, errors="coerce"
)
total_df = total_df.dropna(
    subset=["Area_Name", "Year"] + numeric_features
)

# Save the cleaned subset used in the experiments.
total_df.to_csv(
    OUTPUT_DIR / "cleaned_total_property.csv", index=False
)

print("\nTotal Property rows:", len(total_df))
print("Unique areas:", total_df["Area_Name"].nunique())


# --------------------------------------------------
# 3. UNIT 3 — STANDARDIZE FEATURES
# --------------------------------------------------

X = total_df[numeric_features].clip(lower=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 4. UNIT 3 — K-MEANS CLUSTERING
# --------------------------------------------------

with mlflow.start_run(run_name="Unit3_KMeans"):
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("n_clusters", 3)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("silhouette_score", float(score))
    
    # FIXED: Replaced 'name' with 'artifact_path'
    mlflow.sklearn.log_model(
        kmeans, artifact_path="kmeans_model"
    )

    clustered = total_df.copy()
    clustered["KMeans_Cluster"] = labels
    clustered.to_csv(
        OUTPUT_DIR / "kmeans_clusters.csv", index=False
    )

    print("\nK-Means silhouette score:", round(score, 4))


# --------------------------------------------------
# 5. UNIT 3 — GAUSSIAN MIXTURE MODEL (GMM + EM)
# --------------------------------------------------

with mlflow.start_run(run_name="Unit3_GMM_EM"):
    gmm = GaussianMixture(
        n_components=3,
        random_state=42,
        n_init=5
    )

    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)

    gmm_silhouette = silhouette_score(
        X_scaled, gmm_labels
    )

    mlflow.log_param("algorithm", "GaussianMixture")
    mlflow.log_param("n_components", 3)
    mlflow.log_param("initializations", 5)
    mlflow.log_metric("bic", float(gmm.bic(X_scaled)))
    mlflow.log_metric("aic", float(gmm.aic(X_scaled)))
    mlflow.log_metric(
        "silhouette_score", float(gmm_silhouette)
    )
    
    # FIXED: Replaced 'name' with 'artifact_path'
    mlflow.sklearn.log_model(
        gmm, artifact_path="gmm_model"
    )

    gmm_result = total_df.copy()
    gmm_result["GMM_Cluster"] = gmm_labels
    gmm_result.to_csv(
        OUTPUT_DIR / "gmm_clusters.csv", index=False
    )

    print("GMM BIC:", round(gmm.bic(X_scaled), 2))
    print("GMM silhouette:", round(gmm_silhouette, 4))


# --------------------------------------------------
# 6. UNIT 3 — PCA
# --------------------------------------------------

with mlflow.start_run(run_name="Unit3_PCA"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    mlflow.log_param("algorithm", "PCA")
    mlflow.log_param("n_components", 2)
    mlflow.log_metric(
        "explained_variance_ratio",
        float(pca.explained_variance_ratio_.sum())
    )

    pca_df = pd.DataFrame({
        "Area_Name": total_df["Area_Name"].values,
        "Year": total_df["Year"].values,
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "KMeans_Cluster": labels
    })
    pca_df.to_csv(
        OUTPUT_DIR / "pca_coordinates.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, s=25
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Property Crime Statistics")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=150)
    plt.close()

    mlflow.log_artifact(
        str(OUTPUT_DIR / "pca_clusters.png")
    )

    print(
        "PCA explained variance:",
        round(pca.explained_variance_ratio_.sum(), 4)
    )


# --------------------------------------------------
# 7. UNIT 4 — BUILD NATIONAL YEARLY SERIES
# --------------------------------------------------

annual = (
    total_df.groupby("Year", as_index=False)[numeric_features]
    .sum()
    .sort_values("Year")
)

annual.to_csv(
    OUTPUT_DIR / "national_annual_series.csv", index=False
)

print("\nNational annual series:\n", annual.to_string(index=False))

annual["Log_Cases_Stolen"] = np.log1p(
    annual["Cases_Property_Stolen"]
)


# --------------------------------------------------
# 8. UNIT 4 — MARKOV MODEL
# --------------------------------------------------

annual["Activity_State"] = pd.qcut(
    annual["Cases_Property_Stolen"].rank(method="first"),
    q=3,
    labels=["Low", "Medium", "High"]
)

states = ["Low", "Medium", "High"]
state_to_id = {state: i for i, state in enumerate(states)}
state_ids = annual["Activity_State"].astype(str).map(
    state_to_id
).to_numpy()

transition_counts = np.zeros((3, 3), dtype=float)

for previous_state, next_state in zip(
    state_ids[:-1], state_ids[1:]
):
    transition_counts[previous_state, next_state] += 1

transition_probabilities = (
    transition_counts + 1
) / (transition_counts.sum(axis=1, keepdims=True) + 3)

transition_df = pd.DataFrame(
    transition_probabilities,
    index=states,
    columns=states
)
transition_df.to_csv(
    OUTPUT_DIR / "markov_transition_matrix.csv"
)

with mlflow.start_run(run_name="Unit4_Markov"):
    mlflow.log_param("algorithm", "MarkovChain")
    mlflow.log_param("number_of_states", 3)
    mlflow.log_param("state_definition", "Yearly count tertiles")
    mlflow.log_artifact(
        str(OUTPUT_DIR / "markov_transition_matrix.csv")
    )

print("\nMarkov transition probabilities:\n")
print(transition_df.round(3))


# --------------------------------------------------
# 9. UNIT 4 — HIDDEN MARKOV MODEL (HMM)
# --------------------------------------------------

sequence = annual[["Log_Cases_Stolen"]].to_numpy()

try:
    with mlflow.start_run(run_name="Unit4_HMM"):
        hmm = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=200,
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(sequence)

        hidden_states = hmm.predict(sequence)
        log_likelihood = hmm.score(sequence)

        annual["HMM_Hidden_State"] = hidden_states
        annual.to_csv(
            OUTPUT_DIR / "annual_with_hmm_states.csv",
            index=False
        )

        mlflow.log_param("algorithm", "GaussianHMM")
        mlflow.log_param("hidden_states", 3)
        mlflow.log_param("iterations", 200)
        mlflow.log_metric(
            "sequence_log_likelihood",
            float(log_likelihood)
        )
        mlflow.log_artifact(
            str(OUTPUT_DIR / "annual_with_hmm_states.csv")
        )

        print("\nHMM log-likelihood:", round(log_likelihood, 4))
        print("HMM hidden states:", hidden_states.tolist())

except Exception as error:
    print("\nHMM experiment could not finish:", error)
    print("Try fewer hidden states or check the hmmlearn installation.")


# --------------------------------------------------
# 10. VISUALIZE THE YEARLY TREND
# --------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    annual["Year"],
    annual["Cases_Property_Stolen"],
    marker="o",
    label="Property cases stolen"
)
plt.plot(
    annual["Year"],
    annual["Cases_Property_Recovered"],
    marker="s",
    label="Property cases recovered"
)
plt.xlabel("Year")
plt.ylabel("Number of cases")
plt.title("Indian Property Crime Statistics (2001–2010)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "annual_crime_trend.png", dpi=150
)
plt.close()

print("\nCompleted. Check the outputs/ folder and MLflow UI.")


# --------------------------------------------------
# 11. UNIT 5 — RANDOM FOREST REGRESSION
# --------------------------------------------------
#
# Why Random Forest?
# This is a small, tabular state/year dataset. Random Forest is a
# robust baseline for nonlinear relationships and needs less tuning
# than boosting models.
#
# To make this a time-aware experiment, predict each area's current
# year's stolen-case count using previous-year values only. The split
# is chronological: train on years <= 2008 and test on years >= 2009.
# --------------------------------------------------

print("\n" + "=" * 65)
print("UNIT 5 — RANDOM FOREST REGRESSION")
print("=" * 65)

# Create one-step lag features within each area. This prevents the model
# from using same-year stolen-property values to predict same-year cases.
time_df = total_df.sort_values(["Area_Name", "Year"]).copy()
time_df["Previous_Year_Cases_Stolen"] = (
    time_df.groupby("Area_Name")["Cases_Property_Stolen"].shift(1)
)
time_df["Previous_Year_Cases_Recovered"] = (
    time_df.groupby("Area_Name")["Cases_Property_Recovered"].shift(1)
)
time_df["Previous_Year_Value_Stolen"] = (
    time_df.groupby("Area_Name")["Value_of_Property_Stolen"].shift(1)
)
time_df["Previous_Year_Value_Recovered"] = (
    time_df.groupby("Area_Name")["Value_of_Property_Recovered"].shift(1)
)

target_column = "Cases_Property_Stolen"
feature_columns = [
    "Area_Name",
    "Year",
    "Previous_Year_Cases_Stolen",
    "Previous_Year_Cases_Recovered",
    "Previous_Year_Value_Stolen",
    "Previous_Year_Value_Recovered",
]

model_df = time_df[feature_columns + [target_column]].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=feature_columns + [target_column])

# Chronological holdout: use earlier years to predict later years.
train_mask = model_df["Year"] <= 2008
test_mask = model_df["Year"] >= 2009
X_train = model_df.loc[train_mask, feature_columns]
y_train = model_df.loc[train_mask, target_column]
X_test = model_df.loc[test_mask, feature_columns]
y_test = model_df.loc[test_mask, target_column]

if len(X_train) < 10 or len(X_test) < 1:
    print(
        "Unit 5 skipped: not enough rows after creating lag features "
        "for the chronological train/test split."
    )
else:
    numeric_columns = [
        "Year",
        "Previous_Year_Cases_Stolen",
        "Previous_Year_Cases_Recovered",
        "Previous_Year_Value_Stolen",
        "Previous_Year_Value_Recovered",
    ]
    categorical_columns = ["Area_Name"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_columns),
        ("categorical", categorical_transformer, categorical_columns),
    ])

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ])

    with mlflow.start_run(run_name="Unit5_Random_Forest"):
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan")

        mlflow.log_param("algorithm", "RandomForestRegressor")
        mlflow.log_param("target", target_column)
        mlflow.log_param("train_years", "2002-2008")
        mlflow.log_param("test_years", "2009-2010")
        mlflow.log_param("split_type", "chronological")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        if np.isfinite(r2):
            mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            model_pipeline,
            artifact_path="random_forest_model",
            # Safe here because this model was trained by this script.
            # Trust only the specific scikit-learn tree types needed.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    comparison_df = pd.DataFrame([{
        "Algorithm": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Train_Years": "2002-2008",
        "Test_Years": "2009-2010",
    }])
    comparison_path = OUTPUT_DIR / "unit5_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    predictions_df = model_df.loc[test_mask, [
        "Area_Name", "Year", target_column
    ]].copy()
    predictions_df = predictions_df.rename(
        columns={target_column: "Actual_Cases_Property_Stolen"}
    )
    predictions_df["Random_Forest_Prediction"] = predictions
    predictions_df.to_csv(
        OUTPUT_DIR / "unit5_test_predictions.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    plt.bar(["Random Forest"], [rmse])
    plt.ylabel("RMSE (lower is better)")
    plt.title("Unit 5 — Random Forest Test RMSE")
    plt.tight_layout()
    comparison_plot = OUTPUT_DIR / "unit5_model_comparison.png"
    plt.savefig(comparison_plot, dpi=150)
    plt.close()

    with mlflow.start_run(run_name="Unit5_Random_Forest_Results"):
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(OUTPUT_DIR / "unit5_test_predictions.csv"))
        mlflow.log_artifact(str(comparison_plot))

    print(f"Random Forest MAE:  {mae:.3f}")
    print(f"Random Forest RMSE: {rmse:.3f}")
    if np.isfinite(r2):
        print(f"Random Forest R2:   {r2:.4f}")
    else:
        print("Random Forest R2:   not calculated (fewer than 2 test rows)")
    print("Training years: 2002-2008 | Testing years: 2009-2010")
    print("\nSaved Unit 5 outputs:")
    print(" -", comparison_path)
    print(" -", OUTPUT_DIR / "unit5_test_predictions.csv")
    print(" -", comparison_plot)

