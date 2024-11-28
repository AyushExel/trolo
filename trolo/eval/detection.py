import torch
from trolo.trainers import DetectionTrainer
from trolo.utils.smart_defaults import infer_pretrained_model

def evaluate(model, device=None, **overrides):
    model = infer_pretrained_model(model)
    ckpt = torch.load(model, map_location=device)
    config = ckpt['cfg']
    del ckpt

    trainer = DetectionTrainer(config, pretrained_model=model, **overrides)
    trainer.val(device=device)

