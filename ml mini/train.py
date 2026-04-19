"""Train decision tree regressor and save artifact for the web app."""
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACT_PATH = Path(__file__).parent / "model.joblib"


def main():
    bunch = fetch_california_housing()
    X, y = bunch.data, bunch.target  # sklearn Bunch
    feature_names = list(bunch.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=12, min_samples_leaf=8, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"Test MAE: {mae:.4f} (100k USD)")
    print(f"Test R2:  {r2:.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_description": "Median house value in $100,000s",
        },
        ARTIFACT_PATH,
    )
    print(f"Saved: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
