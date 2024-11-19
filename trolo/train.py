import os
from pathlib import Path

def train_model(
    model_size: str = "l",
    dataset_type: str = "coco",
    config_path: str = None,
    resume_path: str = None,
    output_dir: str = None,
    use_amp: bool = True,
    num_gpus: int = 4,
    seed: int = 0,
    save_logs: bool = True
) -> None:
    """
    Train a D-FINE model with simplified parameters.
    
    Args:
        model_size: Size of the model ('n', 's', 'm', 'l', 'x')
        dataset_type: Type of dataset ('coco', 'obj365', 'obj2coco', 'custom')
        config_path: Optional custom config path. If None, uses default based on model_size and dataset_type
        resume_path: Path to checkpoint to resume training from
        output_dir: Directory to save outputs. If None, creates one based on model_size and dataset_type
        use_amp: Whether to use automatic mixed precision training
        num_gpus: Number of GPUs to use for training
        seed: Random seed for reproducibility
        save_logs: Whether to save training logs to a file
    """
    # Validate inputs
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        raise ValueError("model_size must be one of: n, s, m, l, x")
    
    if dataset_type not in ['coco', 'obj365', 'obj2coco', 'custom']:
        raise ValueError("dataset_type must be one of: coco, obj365, obj2coco, custom")

    # Set default config path if not provided
    if config_path is None:
        if dataset_type == 'obj365':
            config_path = f"configs/dfine/objects365/dfine_hgnetv2_{model_size}_{dataset_type}.yml"
        elif dataset_type == 'obj2coco':
            config_path = f"configs/dfine/objects365/dfine_hgnetv2_{model_size}_{dataset_type}.yml"
        elif dataset_type == 'custom':
            config_path = f"configs/dfine/custom/dfine_hgnetv2_{model_size}_custom.yml"
        else:
            config_path = f"configs/dfine/dfine_hgnetv2_{model_size}_coco.yml"

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"output/{model_size}_{dataset_type}"

    # Construct the training command
    gpu_devices = ','.join(str(i) for i in range(num_gpus))
    base_cmd = f"CUDA_VISIBLE_DEVICES={gpu_devices} torchrun --master_port=7777 --nproc_per_node={num_gpus}"
    train_cmd = f"{base_cmd} train.py -c {config_path} --output-dir {output_dir} --seed={seed}"

    if use_amp:
        train_cmd += " --use-amp"
    
    if resume_path:
        train_cmd += f" -r {resume_path}"

    # Add log redirection if requested
    if save_logs:
        log_file = f"{output_dir}/{model_size}_{dataset_type}.txt"
        train_cmd += f" &> \"{log_file}\" 2>&1"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Execute training command
    print(f"Starting training with command:\n{train_cmd}")
    result = os.system(train_cmd)

    # Handle training failure and automatic resume
    if result != 0:
        print("First training attempt failed, attempting to resume...")
        resume_path = f"{output_dir}/last.pth"
        resume_cmd = f"{base_cmd} train.py -c {config_path} --output-dir {output_dir} --seed={seed}"
        
        if use_amp:
            resume_cmd += " --use-amp"
        
        resume_cmd += f" -r {resume_path}"
        
        if save_logs:
            log_file = f"{output_dir}/{model_size}_{dataset_type}_resume.txt"
            resume_cmd += f" &> \"{log_file}\" 2>&1"
        
        os.system(resume_cmd)
