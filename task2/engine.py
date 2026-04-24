import torch

def train_one_epoch(model, loader, optimizer, device, use_focal=False):
    model.train()
    total_loss = 0

    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)

        if use_focal:
            loss = loss_dict["loss_box_reg"] + loss_dict["loss_objectness"]
        else:
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)