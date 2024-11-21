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
@click.option('--config', '-c', type=str, help='Config name or path')
@click.option('--model', '-m', type=str, help='Model name or path')
@click.option('--dataset', '-d', type=str, help='Dataset name or path')
@click.option('--pretrained', '-p', type=str, help='Pretrained model name or path')
@click.option('--device', type=str, default=None, help='Device to run on (cpu/cuda)')
@click.argument('args', nargs=-1)
def train(config, model, dataset, pretrained, device, args):
    """Train a model using either combined config or separate model/dataset configs"""
    # Convert args to kwargs
    kwargs = dict(arg.split('=') for arg in args if '=' in arg)
    
    # Initialize trainer
    trainer = DetectionTrainer(
        config=config,
        model=model,
        dataset=dataset,
        pretrained_model=pretrained,
        device=device or infer_device(),
        **kwargs
    )
    
    # Start training
    trainer.fit()

def main():
    cli()

if __name__ == '__main__':
    main()