import torch
from torch.utils.data import DataLoader
from dataset import BirdsAIDataset
from model import load_model

def collate_fn(x): return tuple(zip(*x))

def run(data_dir):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    ds = BirdsAIDataset(data_dir)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    model = load_model().to(device)
    model.eval()

    with torch.no_grad():
        for imgs, _ in dl:
            imgs = [i.to(device) for i in imgs]
            outputs = model(imgs)
            print(outputs)
            break

if __name__ == "__main__":
    import sys
    run(sys.argv[1])