import pandas as pd

def compare_prediction_csv(file1, file2):
    # Load CSVs
    df1 = pd.read_csv(file1, sep="\t")
    df2 = pd.read_csv(file2, sep="\t")

    # Check same shape
    if df1.shape != df2.shape:
        print(f"❗ Files have different shapes: {df1.shape} vs {df2.shape}")
        return

    # Check same IDs
    if not (df1["id"] == df2["id"]).all():
        print(f"❗ Files have mismatched IDs. Cannot compare predictions safely.")
        return

    # Compare predictions
    total_positions = len(df1)
    same = (df1["secondary_structure"] == df2["secondary_structure"]).sum()
    different = total_positions - same

    if different == 0:
        print(f"✅ Predictions are identical in both files ({total_positions} positions).")
    else:
        print(f"⚠ Predictions differ at {different}/{total_positions} positions ({100 * different / total_positions:.4f}%).")

# Example usage:
compare_prediction_csv("predictions_run1.csv", "predictions_run2.csv")
