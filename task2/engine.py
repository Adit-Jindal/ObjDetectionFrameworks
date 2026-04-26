import torch

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    print(len(loader))

    for imgs, targets in loader:
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

    return total_loss / len(loader)