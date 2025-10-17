import os
import argparse
from random import randint
import uuid

from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from models import build_model

import wandb


torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    # print("xs: ", xs.shape, "ys: ", ys.shape)
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    print("n_dims: ", n_dims)
    bsize = args.training.batch_size
    
    # args.training.data is "gaussian" or "ar_warmup"
    data_sampler = get_data_sampler(args.training.data, n_dims=args.training.curriculum.dims.start, lag=task_kwargs["lag"])
    task_kwargs = {}
    if hasattr(args.training, 'task_kwargs') and args.training.task_kwargs:
        if hasattr(args.training.task_kwargs, '__dict__'):
            task_kwargs = args.training.task_kwargs.__dict__
        else:
            task_kwargs = args.training.task_kwargs
    
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        task = task_sampler(**task_sampler_args)
        # print(f"Shape of xs: {xs.shape}")
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def deep_merge(dict1, dict2):
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'inherit' in config:
        base_config = {}
        for inherit_file in config['inherit']:
            inherit_path = os.path.join(os.path.dirname(config_path), inherit_file)
            with open(inherit_path, 'r') as f:
                inherited = yaml.safe_load(f)
                base_config = deep_merge(base_config, inherited)
        
        # Remove inherit key and merge with base config
        inherit_key = config.pop('inherit', None)
        config = deep_merge(base_config, config)
    
    # Apply default values for missing keys
    defaults = {
        'training': {
            'num_tasks': None,
            'num_training_examples': None,
            'resume_id': None
        },
        'wandb': {
            'entity': None,
            'name': None,
            'notes': ''
        }
    }
    
    config = deep_merge(defaults, config)
    return config


def create_args_namespace(config):
    """Convert config dict to namespace object"""
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, Args(value))
                else:
                    setattr(self, key, value)
        
        def __repr__(self):
            return str(self.__dict__)
    
    return Args(config)


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        try:
            wandb.init(
                dir=args.out_dir,
                project=args.wandb.project,
                entity=args.wandb.entity,
                config=args.__dict__,
                notes=args.wandb.notes,
                name=args.wandb.name,
                resume=True,
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")
            args.test_run = True

    model = build_model(args.model)
    print(args)
    print(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print(f"Using device: {device}")

    train(model, args, device)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train in-context learning model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = load_config(cmd_args.config)
    args = create_args_namespace(config)
    
    assert args.model.family in ["gpt2", "lstm"]

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    main(args)
