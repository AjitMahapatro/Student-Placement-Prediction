import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("student_placement_dataset_updated.csv")

X = df.drop("status", axis=1)
y = df["status"]

categorical_cols = ["gender", "branch"]
numerical_cols = [
    "cgpa", "internships", "aptitude_score",
    "communication_score", "projects", "backlogs"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss"
        ))
    ]
)

model.fit(X, y)

joblib.dump(model, "model/placement_model.pkl")

print("Model trained and saved successfully.")
