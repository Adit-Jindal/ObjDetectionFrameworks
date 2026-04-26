import torch

def load_model(num_classes=2):
    model = torch.hub.load(
        "fundamentalvision/Deformable-DETR",
        "deformable_detr_resnet50",
        pretrained=True
    )

    model.class_embed = torch.nn.Linear(model.class_embed.in_features, num_classes)
    return model