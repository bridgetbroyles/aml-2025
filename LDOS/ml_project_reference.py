# Instructor Key: Machine Learning for Systems Project
# -----------------------------------------------------
# This script covers loading, processing, training, and evaluating a perceptron model
# using the 100 .parquet files in the `rainsong_labeled` directory.

import polars as pl
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from library import process_block_io_flags  # custom preprocessing function

# STEP 1: Load all .parquet files from the folder
df = pl.read_parquet("rainsong_labeled/*.parquet")
print("Loaded shape:", df.shape)

# STEP 2 & 3: Drop extra target columns and quantize string flags
df = df.drop([
    'collection_id',
    'block_io_latency_us',
    'block_latency_us',
    'measured_latency_us',
    'label85',
    'label95',
    'finish_ts_uptime_us'
])
df = process_block_io_flags(df)
print("After processing flags:", df.columns)

# STEP 4: Exploratory Data Analysis (EDA)
print("Target value counts (label90):")
print(df["label90"].value_counts())

# Optional: show basic stats
print(df.describe())

# STEP 5: Filter to 'Read' rows and split into train/test (80/20)
df = df.filter(pl.col("Read"))
df = df.with_row_index()  # for reproducible splitting

train_df = df.sample(fraction=0.8, with_replacement=False, seed=42)
test_df = df.filter(~pl.col("index").is_in(train_df["index"]))

train_df = train_df.drop("index")
test_df = test_df.drop("index")

# Confirm shape
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# STEP 6: Train a simple perceptron (logistic regression)
X_train = train_df.drop("label90").to_numpy()
y_train = train_df["label90"].to_numpy()

X_test = test_df.drop("label90").to_numpy()
y_test = test_df["label90"].to_numpy()

# Perceptron (no hidden layers)
model = MLPClassifier(
    hidden_layer_sizes=(),  # no hidden layers = logistic regression
    activation='relu',      # has no effect without hidden layers
    solver='adam',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained.")

# STEP 7: Evaluate with confusion matrix and metrics
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# STEP 8: Read documentation for improvements
# (Done outside of code: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

# STEP 9: Adjust model - add 1 hidden layer with 50 nodes
model_improved = MLPClassifier(
    hidden_layer_sizes=(50,),  # one hidden layer
    activation='relu',
    solver='adam',
    alpha=0.001,               # small L2 regularization
    max_iter=500,
    random_state=42
)
model_improved.fit(X_train, y_train)
y_pred_improved = model_improved.predict(X_test)

print("\n[Improved Model Results]")
print(confusion_matrix(y_test, y_pred_improved))
print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_improved):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_improved):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_improved):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_improved):.4f}")

# STEP 10: Continue tuning (students should try different hidden layers, activations, etc.)
# Example:
# hidden_layer_sizes=(50, 25)
# activation='tanh'
# solver='sgd'
# learning_rate='adaptive'

# End of instructor key
