import torch
from torch.utils.data import DataLoader
from dataset import BirdsAIDataset
from model import load_model
from engine import train_one_epoch
from utils import freeze_all

def collate_fn(x): return tuple(zip(*x))

def run(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BirdsAIDataset(data_dir)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = load_model().to(device)
    freeze_all(model)

    for name, p in model.named_parameters():
        if "encoder" in name:
            p.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    for epoch in range(20):
        loss = train_one_epoch(model, dl, optimizer, model.criterion, device)
        print("Epoch", epoch, loss)

    torch.save(model.state_dict(), "encoder_ft.pth")

if __name__ == "__main__":
    import sys
    run(sys.argv[1])