import csv
import os

def log_metrics(log_file, epoch, loss):
    exists = os.path.exists(log_file)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(["epoch", "loss"])

        writer.writerow([epoch, loss])