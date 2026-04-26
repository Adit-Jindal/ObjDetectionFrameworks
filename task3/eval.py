import torch
from torch.utils.data import DataLoader
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from dataset import BirdsAIDataset, collate_fn

def evaluate(data_dir, ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device))

    model.to(device)
    model.eval()

    dataset = BirdsAIDataset(data_dir, processor)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            print(outputs)
            break