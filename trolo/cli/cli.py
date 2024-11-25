import click
from pathlib import Path
from trolo.trainers.detection import DetectionTrainer
from trolo.utils.smart_defaults import infer_device
from trolo.loaders.maps import get_model_config_path

@click.group()
def cli():
    """CLI tool for D-FINE model training and inference"""
    pass

@cli.command()
@click.option('--config', '-c', type=str, default=None, help='Config name or path')
@click.option('--model', '-m', type=str, default=None, help='Model name or path')
@click.option('--dataset', '-d', type=str, default=None, help='Dataset name or path')
@click.option('--pretrained', '-p', type=str, default=None, help='Pretrained model name or path')
@click.option('--device', '-dev', type=str, default=None, help='Device specification')
def train(config, model, dataset, pretrained, device, overrides={}):
    """Train a model using either combined config or separate model/dataset configs"""

    # Initialize trainer
    trainer = DetectionTrainer(
        config=config,
        model=model,
        dataset=dataset,
        pretrained_model=pretrained,
        overrides=overrides
    )
    
    # Start training
    trainer.fit(device=device)

def main():
    cli()

if __name__ == '__main__':
    main()