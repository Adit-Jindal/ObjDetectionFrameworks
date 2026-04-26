import torch
from torch.utils.data import DataLoader
from dataset import BirdsAIDataset
from model import get_model
from engine import train_one_epoch
from utils import log_metrics

def collate_fn(batch):
    return tuple(zip(*batch))


def train(data_dir, experiment="ce"):
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    train_ds = BirdsAIDataset(data_dir)
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn)

    model = get_model()
    model.to(device)

    print("Model instatiated")

    optimizer = torch.optim.SGD([p for p in model.parameters()if p.requires_grad], lr=0.005, momentum=0.9)

    log_file = f"task2/runs/logs/{experiment}.csv"

    for epoch in range(20):
        print(f'Starting epoch {epoch}...')
        model.train()
        total_loss = 0
        print(len(train_loader))

        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)

            loss = sum((loss_dict.values()))
            if torch.isnan(loss):
                print(f'NaN found')
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Training with {loss}")

        loss = total_loss / len(train_loader)

        log_metrics(log_file, epoch, loss)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f"task2/runs/models/{experiment}_best.pth")
        print(f"Epoch {epoch}: Loss {loss:.4f}")


if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])