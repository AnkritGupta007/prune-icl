import pandas as pd

df = pd.read_csv("manifests/phase1_minimal.csv")

print("total rows:", len(df))
print("\nphase counts:")
print(df["phase"].value_counts().sort_index())

print("\nphase1 task counts:")
print(df[df["phase"]=="phase1"]["task"].value_counts())

print("\nphase1 method counts:")
print(df[df["phase"]=="phase1"]["method"].value_counts())

print("\nenabled rows:", int(df["enabled"].sum()))

assert len(df) == 30, f"Expected 30 rows total, got {len(df)}"
assert len(df[df['phase']=="phase1"]) == 25, "Phase 1 must be exactly 25 runs"
assert len(df[df['phase']=="phase0"]) == 5, "Phase 0 must be 5 runs"

print("\nmanifest check passed")
