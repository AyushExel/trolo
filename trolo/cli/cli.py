import click
from pathlib import Path
from trolo.trainers.detection import DetectionTrainer
from trolo.inference.detection import DetectionPredictor
from trolo.utils.smart_defaults import infer_device, infer_input_path, infer_pretrained_model, infer_output_path, DEFAULT_MODEL
from trolo.eval.detection import evaluate as eval_detection

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
@click.option('--batch-size', '-bs', type=int, default=None, help='Batch size')
def train(config, model, dataset, pretrained, device, batch_size):
    """Train a model using either combined config or separate model/dataset configs"""

    overrides = {}
    if batch_size:
        overrides['batch_size'] = batch_size

    # Initialize trainer
    trainer = DetectionTrainer(
        config=config,
        model=model,
        dataset=dataset,
        pretrained_model=pretrained,
        **overrides
    )
    
    # Start training
    trainer.fit(device=device)

@cli.command()
@click.option('--model', '-m', type=str, default=DEFAULT_MODEL, help='Model name or path')
@click.option('--input', '-i', type=str, default=None, help='Input image or directory path')
@click.option('--output', '-o', type=str, default=None, help='Output directory path')
@click.option('--device', '-dev', type=str, default=None, help='Device specification')
@click.option('--conf-thresh', '-ct', type=float, default=None, help='Confidence threshold')
@click.option('--save', is_flag=True, default=False, help='Save visualization results')
@click.option('--no-show', is_flag=True, default=False, help='Disable result visualization')
def predict(model, input, output, device, conf_thresh, save, no_show):
    """Run inference on images using a trained model"""
    
    # Initialize predictor with smart defaults
    predictor = DetectionPredictor(
        model=infer_pretrained_model(model),
        device=device or infer_device()
    )
    
    # Infer paths and parameters with smart defaults
    output_path = Path(infer_output_path(output)) if output else None
    conf_thresh = conf_thresh or 0.5
    
    # Run visualization
    predictor.visualize(
        input=input,
        show=not no_show,  # Show by default unless --no-show is used
        save=save,
        save_dir=output_path,
        conf_threshold=conf_thresh
    )

@cli.command()
@click.option('--model', '-m', type=str, default=None, help='Model name or path')
@click.option('--device', '-dev', type=str, default=None, help='Device specification')
@click.option('--batch-size', '-bs', type=int, default=None, help='Batch size')
def eval(model, device, batch_size):
    """Train a model using either combined config or separate model/dataset configs"""
    
    overrides = {}
    if batch_size:
        overrides['batch_size'] = batch_size

    # Support DDP evaluation
    device = device or infer_device()
    eval_detection(model, device=device, **overrides)

def main():
    cli()

if __name__ == '__main__':
    main()