import os
import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid blocking in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from catboost import CatBoostClassifier


def train_model(
    data_path: str = "tess_training_data.csv",
    save_fig: str = "confusion_matrix.png",
    save_model_path: str = "catboost_exoplanet.cbm",
    show_plot: bool = False,
    verbose: int = 100,
):
    """
    Train the CatBoost-based exoplanet classifier with robust preprocessing and evaluation.

    Parameters
    - data_path: Path to the CSV file containing training data.
    - save_fig: Path to save the confusion matrix figure.
    - save_model_path: Path to save the trained CatBoost model.
    - show_plot: If True, attempt to display the plot (may be ignored in headless).
    - verbose: Verbosity level for CatBoost training logging.
    """
    print("1. Loading and Validating Data...")

    # Validate data path
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Ensure the CSV exists and the path is correct."
        )

    # Load the CSV
    df = pd.read_csv(data_path)

    # Validate required columns
    if "label" not in df.columns:
        raise ValueError("Required column 'label' not found in dataset.")

    # Extract and map labels BEFORE any imputation on features
    label_mapping = {"KP": 1, "CP": 1, "FP": 0}
    y = df["label"].map(label_mapping)

    # Drop any rows where the label wasn't recognized just to be safe
    valid_mask = y.notna()
    if valid_mask.sum() != len(df):
        print(
            f"   Warning: Dropping {len(df) - valid_mask.sum()} rows with unrecognized/NaN labels."
        )
    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    # Drop identifier columns from features if present
    drop_cols = [c for c in ["label", "tic_id"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Identify expected feature groups
    flux_cols = [c for c in X.columns if c.startswith("f_")]  # Flux data
    cc_cols = [c for c in X.columns if c.startswith("cc_")]  # Column centroids
    cr_cols = [c for c in X.columns if c.startswith("cr_")]  # Row centroids

    # Sanity warnings if expected groups are missing
    if not flux_cols:
        print("   Warning: No flux feature columns found (prefix 'f_').")
    if not cc_cols:
        print("   Warning: No column-centroid feature columns found (prefix 'cc_').")
    if not cr_cols:
        print("   Warning: No row-centroid feature columns found (prefix 'cr_').")

    print("2. Preprocessing Features (imputation + normalization)...")

    # Impute gaps (NaNs) due to quality masking using forward-fill and backward-fill
    # Apply ONLY to features, not labels or identifiers
    X = X.ffill(axis=1).bfill(axis=1)

    # Normalize flux: divide each row by its median, protect against zero/NaN division
    if flux_cols:
        flux_median = X[flux_cols].median(axis=1).replace(0, np.nan)
        X[flux_cols] = X[flux_cols].div(flux_median, axis=0)

    # Center centroids by row-wise mean
    if cc_cols:
        X[cc_cols] = X[cc_cols].sub(X[cc_cols].mean(axis=1), axis=0)
    if cr_cols:
        X[cr_cols] = X[cr_cols].sub(X[cr_cols].mean(axis=1), axis=0)

    # Scale centroids by row-wise std (to emphasize small movements), avoid div-by-zero
    if cc_cols:
        cc_std = X[cc_cols].std(axis=1).replace(0, np.nan).add(1e-8)
        X[cc_cols] = X[cc_cols].div(cc_std, axis=0)
    if cr_cols:
        cr_std = X[cr_cols].std(axis=1).replace(0, np.nan).add(1e-8)
        X[cr_cols] = X[cr_cols].div(cr_std, axis=0)

    # Post-processing checks/logging
    print(f"   Final feature count: {X.shape[1]}")
    class_counts = y.value_counts().to_dict()
    print(f"   Class distribution (after cleaning): {class_counts}")

    print("3. Splitting Data (80% Train, 10% Validation, 10% Test)...")

    # Helper function to try stratified split and fallback if needed
    def safe_train_val_test_split(Xd, yd, random_state=42):
        # Try stratified split
        try:
            X_train_, X_temp_, y_train_, y_temp_ = train_test_split(
                Xd, yd, test_size=0.20, random_state=random_state, stratify=yd
            )
            X_val_, X_test_, y_val_, y_test_ = train_test_split(
                X_temp_, y_temp_, test_size=0.50, random_state=random_state, stratify=y_temp_
            )
        except ValueError as e:
            print(
                f"   Warning: Stratified split failed ({e}). Falling back to non-stratified split."
            )
            X_train_, X_temp_, y_train_, y_temp_ = train_test_split(
                Xd, yd, test_size=0.20, random_state=random_state, stratify=None
            )
            X_val_, X_test_, y_val_, y_test_ = train_test_split(
                X_temp_, y_temp_, test_size=0.50, random_state=random_state, stratify=None
            )
        return X_train_, X_val_, X_test_, y_train_, y_val_, y_test_

    X_train, X_val, X_test, y_train, y_val, y_test = safe_train_val_test_split(X, y)

    print(
        f"   Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}"
    )

    print("4. Initializing and Training CatBoost...")
    # Parameters tailored for high-dimensional, small-sample data
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,  # Smaller learning rate prevents overshooting
        depth=4,  # Shallow trees prevent overfitting on small datasets
        l2_leaf_reg=5,  # L2 regularization for many features
        eval_metric="Accuracy",
        random_seed=42,
        verbose=verbose,  # Prints progress every `verbose` steps
    )

    # Fit the model, using the Validation set for Early Stopping
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,  # Stops if val accuracy doesn't improve for 50 rounds
        use_best_model=True,
    )

    print("\n5. Evaluating on Test Data...")
    y_pred = model.predict(X_test)
    # Ensure predictions are 1D integer labels
    y_pred = np.asarray(y_pred).ravel().astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Final Test Accuracy: {acc * 100:.2f}%")

    print("6. Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["False Positive (0)", "Planet (1)"]
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Purples, ax=ax)
    plt.title(f"CatBoost TESS Exoplanet Classifier\nTest Accuracy: {acc * 100:.2f}%")
    plt.tight_layout()

    # Save figure always; optionally show if requested
    try:
        plt.savefig(save_fig, dpi=150)
        print(f"   Confusion matrix saved to: {save_fig}")
    except Exception as e:
        print(f"   Warning: Failed to save figure to {save_fig}: {e}")

    if show_plot:
        # This may still no-op depending on backend, but is user-controlled
        try:
            plt.show()
        except Exception as e:
            print(f"   Warning: Unable to show plot: {e}")
    plt.close(fig)

    # Persist the trained model
    try:
        model.save_model(save_model_path)
        print(f"7. Model saved to: {save_model_path}")
    except Exception as e:
        print(f"   Warning: Failed to save model to {save_model_path}: {e}")

    return model, {
        "accuracy": float(acc),
        "class_distribution": class_counts,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "confusion_matrix_path": save_fig,
        "model_path": save_model_path,
    }


if __name__ == "__main__":
    train_model()
