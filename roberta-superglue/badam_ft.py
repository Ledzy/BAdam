import sys
sys.path.insert(0, "..")

import os
import argparse
import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader
import wandb
import argparse

# DO NOT train embedding
Roberta_base_block_list = [*[[f"encoder.encoder.layer.{i}."] for i in range(11)], 
                           ["encoder.pooler."], ["taskmodels_dict."]]

Roberta_large_block_list = [*[[f"encoder.encoder.layer.{i}."] for i in range(24)], 
                           ["encoder.pooler."], ["taskmodels_dict."]]

# Create an argument parser
parser = argparse.ArgumentParser(description="Argument Parser for jiant")

# Add arguments for the capitalized variables
parser.add_argument("--task_name", type=str, default="boolq", help="Name of the task")
parser.add_argument("--hf_pretrained_model_name", type=str, default="roberta-base", help="Name of the Hugging Face pretrained model")
parser.add_argument("--exp_dir", type=str, default="./content/exp", help="Directory for experiment")
parser.add_argument("--data_dir", type=str, default="./content/exp/tasks", help="Directory for task data")
parser.add_argument("--use_block_optim", action='store_true', help="whether to use block optimizer")
parser.add_argument("--use_sparse_optim", action='store_true', help="whether to use sparse optimizer")
parser.add_argument("--switch_every", type=int, default=10, help="switch block every n steps")
parser.add_argument("--switch_mode", type=str, default="ascending", help="switch mode for block optimizer")
parser.add_argument("--start_block", type=int, default=None, help="start block for block optimizer")
parser.add_argument("--num_train_epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--eval_every_steps", type=int, default=20, help="evaluate every n steps")
parser.add_argument("--train_batch_size", type=int, default=32, help="training batch size")
parser.add_argument("--update_ratio", type=float, default=0.1, help="update parameter ratio for sparse optimizer")
parser.add_argument("--optimizer_type", type=str, default="adam", help="optimizer type")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--train_last_layer", action='store_true', help="train the full last layer through the whole training process")
parser.add_argument("--lora_rank", type=int, default=0, help="rank for lora")

# Parse the arguments
config = parser.parse_args()
assert not (config.use_block_optim and config.use_sparse_optim), "Cannot use both block and sparse optimizers"

if config.use_sparse_optim:
    config.optimizer_type = "sparse_adam"
    print("Using sparse optimizer, changing optimizer type to sparse_adam")

# Assign the parsed arguments to the capitalized variables
MODEL_NAME = config.hf_pretrained_model_name.split("/")[-1]
if config.use_block_optim:
    RUN_NAME = f"{config.task_name}_block_swifreq_{config.switch_every}_{config.switch_mode}"
elif config.use_sparse_optim:
    RUN_NAME = f"{config.task_name}_sparse_swifreq_{config.switch_every}_updateratio_{config.update_ratio}"
elif config.lora_rank > 0:
    RUN_NAME = f"{config.task_name}_lora_rank_{config.lora_rank}"
else:
    RUN_NAME = f"{config.task_name}_full_{config.optimizer_type}"
    
if config.train_last_layer:
    RUN_NAME = RUN_NAME + "train_last"

# TODO: temporary fix
if config.hf_pretrained_model_name == "roberta-base":
    block_prefix_list = Roberta_base_block_list
elif config.hf_pretrained_model_name == "FacebookAI/roberta-large":
    block_prefix_list = Roberta_large_block_list
else:
    block_prefix_list = None

os.makedirs(config.data_dir, exist_ok=True)
os.makedirs(config.exp_dir, exist_ok=True)

# Initialize a new wandb run
wandb.init("SUPERGLUE",
           config=config,
           name=RUN_NAME)

# #Run simple `jiant` pipeline (train and evaluate on MRPC)
args = simple_run.RunConfiguration(
    run_name=RUN_NAME,
    exp_dir=config.exp_dir,
    data_dir=config.data_dir,
    hf_pretrained_model_name_or_path=config.hf_pretrained_model_name,
    tasks=config.task_name,
    train_batch_size=config.train_batch_size,
    num_train_epochs=config.num_train_epochs,
    use_block_optim=config.use_block_optim,
    use_sparse_optim=config.use_sparse_optim,
    block_prefix_list=block_prefix_list,
    switch_every=config.switch_every,
    switch_mode=config.switch_mode,
    start_block=config.start_block,
    eval_every_steps=config.eval_every_steps,
    optimizer_type=config.optimizer_type,
    update_ratio=config.update_ratio,
    learning_rate=config.learning_rate,
    train_last_layer=config.train_last_layer,
    lora_rank=config.lora_rank,
    write_test_preds=True,
    write_val_preds=True,
)
simple_run.run_simple(args)