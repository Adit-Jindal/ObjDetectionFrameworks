import torch
from torch.utils.data import DataLoader
from transformers import (
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor
)

from dataset import BirdsAIDataset, collate_fn


def freeze_model(model, mode):
    if mode == "full":
        return

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if mode == "decoder" and "model.decoder" in name:
            p.requires_grad = True
        elif mode == "encoder" and "model.encoder" in name:
            p.requires_grad = True


def train(data_dir, mode):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    processor = DeformableDetrImageProcessor.from_pretrained(
        "SenseTime/deformable-detr"
    )

    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    freeze_model(model, mode)
    model.to(device)

    dataset = BirdsAIDataset(data_dir, processor)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    lr = 2e-4 if mode == "full" else 1e-4

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    for epoch in range(4):
        model.train()
        total_loss = 0
        i = 0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            total_loss += loss.item()
            print(f"Training count {i} with loss {loss}")
            i += 1

        print(f"[{mode}] Epoch {epoch}: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), f"{mode}_model.pth")


if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])