import pandas as pd

def compute_scale(area):
    if area < 200:
        return "S"
    elif area < 2000:
        return "M"
    else:
        return "L"

def evaluate_model(model, data_dir):
    # Placeholder — extend this
    print("Running custom evaluation...")

# TODO:
# 1. Read CSV again
# 2. Compute avg area per video (or image group if no video info)
# 3. Assign scale
# 4. Match predictions with GT
# 5. Compute:
#    SA, MA, LA
#    SH, MH, LH
pass
