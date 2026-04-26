import torch

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def freeze_encoder(model):
    for name, p in model.named_parameters():
        p.requires_grad = "encoder" in name

def freeze_decoder(model):
    for name, p in model.named_parameters():
        p.requires_grad = "decoder" in name