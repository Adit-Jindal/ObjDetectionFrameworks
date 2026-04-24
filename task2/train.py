import torch
from torch.utils.data import DataLoader
from dataset import BirdsAIDataset
from model import get_model
from engine import train_one_epoch
from utils import log_metrics

def collate_fn(batch):
    return tuple(zip(*batch))


def train(data_dir, experiment="ce"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BirdsAIDataset(data_dir)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = get_model()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    log_file = f"logs/{experiment}.csv"

    for epoch in range(20):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_focal=(experiment == "fl")
        )

        log_metrics(log_file, epoch, loss)
        print(f"Epoch {epoch}: Loss {loss:.4f}")


if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])