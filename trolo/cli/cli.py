import click
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trolo.train import train_model
from trolo.infer import process_image, process_video, load_model
from trolo.loaders import yaml_utils
from trolo.utils.smart_defaults import infer_pretrained_model, infer_input_path, infer_device

@click.group()
def cli():
    """D-FINE CLI tool for training, testing, and inference"""
    pass

@cli.command()
@click.argument('args', nargs=-1)
@click.option('--model-size', '-m', default='l', type=click.Choice(['n', 's', 'm', 'l', 'x']))
@click.option('--dataset', '-d', default='coco', type=click.Choice(['coco', 'obj365', 'obj2coco', 'custom']))
@click.option('--config', '-c', type=click.Path(exists=True))
@click.option('--resume', '-r', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path())
@click.option('--use-amp/--no-amp', default=True)
@click.option('--num-gpus', default=4, type=int)
@click.option('--seed', default=0, type=int)
@click.option('--save-logs/--no-logs', default=True)
def train(args, model_size, dataset, config, resume, output_dir, use_amp, num_gpus, seed, save_logs):
    """Train a model with specified parameters"""
    # Parse additional arguments
    extra_args = yaml_utils.parse_cli(args)
    
    train_model(
        model_size=model_size,
        dataset_type=dataset,
        config_path=config,
        resume_path=resume,
        output_dir=output_dir,
        use_amp=use_amp,
        num_gpus=num_gpus,
        seed=seed,
        save_logs=save_logs,
        **extra_args
    )

@cli.command()
@click.argument('args', nargs=-1)
@click.option('--model', '-m', type=click.Path(exists=True), default=infer_pretrained_model())
@click.option('--input', '-i', type=click.Path(exists=True), default=infer_input_path())
@click.option('--output', '-o', type=click.Path())
@click.option('--device', '-d', default=infer_device())
@click.option('--format', '-f', type=click.Choice(['torch', 'onnx', 'trt']), default='torch')
@click.option('--show', '-s', is_flag=True, help='Show the output image or video in a window', default=True)
@click.option('--save', '-v', is_flag=True, help='Save the output image or video', default=True)
def infer(args, model, input, output, device, format, show, save):
    """Run inference on images or videos"""
    
    # check if input is a video file, if so then use process_video otherwise process it as image
    # Parse additional arguments
    extra_args = yaml_utils.parse_cli(args)

    # Check if input is video by extension
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    input_path = Path(input)
    is_video = input_path.suffix.lower() in video_exts

    # Load model
    model = load_model(model, format, device)

    if is_video:
        process_video(model, input, device, format, show, save)
    else:
        process_image(model, input, device, format, show, save)
    

def main():
    cli()

if __name__ == '__main__':
    main()