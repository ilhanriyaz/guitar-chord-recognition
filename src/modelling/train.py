import pandas as pd
from xgboost import XGBClassifier

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

chroma_cols = [col for col in train_df.columns if col.startswith("feature_")]
X_train = train_df[chroma_cols].values
y_train = train_df["label"].values
X_test = test_df[chroma_cols].values
y_test = test_df["label"].values

xg_model = XGBClassifier()
