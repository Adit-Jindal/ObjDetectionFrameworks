import csv
import os

def log_metrics(log_file, epoch, loss):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    exists = os.path.exists(log_file)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(["epoch", "loss"])

        writer.writerow([epoch, loss])