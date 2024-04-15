# BAdam

The implementation for [BAdam: A Memory Efficient Full Parameter Training Method for Large Language Models](https://arxiv.org/abs/2404.02827). The paper proposes an algorithm named **BAdam**, which finetunes Llama 2-7b on a single RTX-3090 with Adam's update rule and mixed precision training. The core idea of **BAdam** is to sequentially solve block coordinate sub-problems. From the implementation perspective, the algorithm runs Adam's update on small portition (usually one single transformer layer) of the parameter, thereby requires much less memory in comparison to full parameter Adam finetuning. 

| Method | Minimum Memory | Memory Cost (Llama 2-7b) |
| -------- | -------- | -------- |
| Adam    | $18M$     | 109.4 GB+     |
| **BAdam**    | $2M + \frac{16M}{D}$   | 21.8 GB     |
<!-- | LoRA    | Data     | Data     | -->
**Table 1: Comparison of Methods.** $M$ stands for the number of model's parameters in billion. See Table 4 in paper for detailed analysis on memory consumption.

## Table of Contents
- [Environment Setup](#setup)
- [Usage of BAdam](#how-to-use-badam)
    - [Partition by Module](#partition-by-module)
    - [Partition by Parameter Ratio](#partition-by-parameter-ratio)
- [Paper's Experiment](#how-to-run-the-experiment)
    - [Llama 2-7b on Alpaca-GPT4](#llama-2-7b-on-alpaca-gpt4)
    - [RoBERTa-large on superGLUE](#roberta-large-on-superglue)

## Setup
To install **BAdam** from Pypi, you can run:
```bash
pip install badam
```

You may also choose to build from source by following steps:
```bash
git clone git@github.com:Ledzy/BAdam.git
cd BAdam
pip install -e .
```

For those who are interested in reproducing the results in paper, please follow the steps below to setup environment:
```bash
conda create -n badam python=3.10
conda activate badam
pip install -r requirements.txt
```

## Usage of BAdam

### Partition by Module
To use **BAdam**, one can simply add one line of code that wraps the original optimizer.

```python
from block_optim import BlockOptimizer

# before training, add this line to wrap the original optimizer
optimizer = BlockOptimizer(
    base_optimizer=original_optimizer, # can be any torch.Optimizer
    named_parameters_list=list(model.named_parameters()), 
    switch_block_every=50, # switch to the new block every 50 updates, the $K$ Adam steps in paper
    switch_mode="ascending", # update order of blocks, "ascending" means update the input layers first
    verbose=2 # information level, will print trainable parameters when setting to 2
)
```
The above code automatically creates the block list according to `model.named_parameters`. Specifically, it treates the embedding layer as a single block, each transformer layer as a single block, and group all the other parameters as a single block. For instance, for the Llama2-7b, the block partition will be
```
block 1: model.embed_tokens.
block 2: model.layers.0.
block 3: model.layers.1.
...
block 33: model.layers.31.
block 34: lm_head., model_norm.
```

One can also specify their own block list for the block optimizer. This can be achieved by adjusting the`block_prefix_list` argument. For instance, the following code snippet divide each layer into blocks, which helps further reduce the memory cost:

```python
block_prefix_list = []
for i in range(31):
    layer_prefix = [
        [f"model.layers.{i}.self_attn.q_proj."],
        [f"model.layers.{i}.self_attn.k_proj."],
        [f"model.layers.{i}.self_attn.v_proj."],
        [f"model.layers.{i}.self_attn.o_proj."],
        [f"model.layers.{i}.mlp.gate_proj."],
        [f"model.layers.{i}.mlp.up_proj."],
        [f"model.layers.{i}.mlp.down_proj."],
    ]
    block_prefix_list.extend(layer_prefix)

optimizer = BlockOptimizer(
    base_optimizer=original_optimizer,
    named_parameters_list=list(model.named_parameters_list), 
    switch_block_every=50,
    switch_mode="ascending",
    verbose=2,
    block_prefix_list=block_prefix_list # set the block list
)
```
**Note:**:
* When setting block partition, one should be careful with the downstream task. Some tasks has randomly initialized layers, such as the superGLUE where the `task_dict` and `pooler` layers are randomly initialized. **In this case, make sure to train these layers first, or set it to be trainable through the whole time.** To set modules to be trainable through the training, you can use `active_modules` argument, e.g. set `active_modules=["model.task_dict.", "model.pooler."]` when create the BlockOptimizer. Note that randomly initialized layers are usually the last layer, so updating these layers will not cause large overhead. We suggest to always set the last layer to be trainable when the memory is permitted.
* The parameters that are not included in `block_prefix_list` will be freezed through the whole training procedure.
* When setting prefix, it is suggested to include a `.` at the end. For example, it is preferred to use `model.layers.1.` instead of `model.layers.1`, as the later one includes the layer 10, 11, ..., 19 as well (since they have the same prefix).
* Currently, all the experiments are conducted in a single GPU. Using this code in distributed training may exhibit unpredictable behaviors. For instance, when using pytorch DDP, the reducer for gradient synchronization are created when initializing the DDP optimizer. When switching to block where the reducer are not created, the block will **NOT** be updated as expected. The code for distributed training is under active development.


### Partition by Parameter Ratio
Instead of partitioning block by the model's parameter, an alternative choice is to train all the parameters simultaneously with fixed ratio. For instance, we can train 5% of every parameter. In this sense, the feature extractor of every layer are jointly trained, which may be preferred in certain scenarios. However, training a block consisting of parameters coming from all the transformer layers loses the benefit of time saving of BlockOptimizer

To do this, one can use the `BlockOptimizerRatio`:

```python
from block_optim import BlockOptimizerRatio

optimizer = BlockOptimizerRatio(
    param_groups=param_groups, # param_group of torch.Optimizer, the same as the original optimizer
    named_parameters_list=list(self.model.named_parameters()),
    switch_every=100, # switch to the new block every 100 updates
    update_ratio=0.1, # ratio of trainable weight for each parameter
    mask_mode = "adjacent", # choices: ["adjacent", "scatter"], see Note below for more explanation
    lr=1e-5,
    betas=(0.9, 0.999), # betas for Adam update
    eps=1e-8, # eps of Adam update
)
```
Currently, the `BlockOptimizerRatio` only supports the `Adam` update. The repository is still under active development.

**Note:**
* The `mask_mode` indicates how should the trainable parameter distribute across a parameter. `mask_mode=adjacent` indicates that the trainable parameters are adjacent to each other, while `mask_mode=scatter` indicates that trainable parameters are randomly choosed from the weight. For instance, for a $10 \times 10$ matrix, setting `mask_mode=adjacent` will let parameters of the same row be the same block, and `mask_mode=scatter` means randomly choose 10 trainable parameters from the matrix.
* By default, `BlockOptimizerRatio` doesn't update embedding layer, since in principle the embedding vectors of the tokens that are included in the training samples should be updated, while randomly freeze embedding parameters makes the update imbalanced. One can set `include_embedding=True` to include it for experimental purpose.
* The gradient and optimizer states are stored in sparse tensor format. The update rule is exactly the same as the  `BlockOptimizer`: run Adam update on current active block for `switch_every` steps, and then switch to next block.
* Currently, the operation of sparsifing the gradient causes noticable overhead, which inevitably slow down the training. We leave the acceleration as a future work.

## How to Run Paper's Experiment

### Llama 2-7b on Alpaca-GPT4
Our implementation of finetuning Llama 2 is based on [Llama Factory](https://github.com/hiyouga/LLaMA-Factory). For the experiment of finetuning Llama-2 7b on [Alpaca-GPT4](https://arxiv.org/abs/2304.03277) dataset, first change the working directory to `llama`:
```bash
cd llama-alpaca
```
Here is a sample command for running the code:
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path alpaca_gpt4_en \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type block \
    --output_dir ./outputs/tmp \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 15 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --val_size 500 \
    --eval_steps 20 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --num_train_epochs 32 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 50 \
    --switch_mode ascending \
    --start_block 1 \
    --bf16
```
**Notes on arguments:**
* `--stage`: Currently we only implement the `sft`.
* `--finetuning_type`: Options: (block, full, lora, sparse)
* `--switch_mode`: How to order the block update. Options: (ascending, descending, random).
* `--switch_every`: Switch block frequency.

### RoBERTa-large on superGLUE
Our implementation for finetuning RoBERTa-large on [superGLUE](https://arxiv.org/abs/1905.00537) is based on [jiant](https://github.com/nyu-mll/jiant). To run the code, go to directory `roberta-superglue` first:
```bash
cd roberta-superglue
```
Before training the model, download the dataset using the following bash script. Adjust the script to download the required dataset.
```bash
EXP_DIR=./content/exp

python jiant/scripts/download_data/runscript.py \
    download \
    --tasks copa \
    --output_path ${EXP_DIR}/tasks
```
The finetuning command has the following form:
```bash
CUDA_VISIBLE_DEVICES=0 python badam_ft.py \
    --task_name boolq \
    --num_train_epochs 32 \
    --eval_every_steps 100 \
    --use_block_optim \
    --switch_every 100 \
    --switch_mode ascending \
    --train_batch_size 16 \
    --train_last_layer \
    --hf_pretrained_model_name FacebookAI/roberta-large
```

**Notes on arguments:**
* `--task_name`: Options: boolq, wic, wsc, rte, multirc, copa
* `--use_block_optim`: Whether to use BlockOptimizer or not. Remove this argument leads to full parameter Adam update. Change to `--use_sparse_optim`: to use BlockOptimizerRatio.
* `--train_last_layer`: Whether to train the last layer through the finetuning. For the superGLUE task, the last layer is randomly initialized and thereby needs to be trained first or being trainable through the whole training.

## Change log
[24/04/15] Package BAdam in Pypi. Remove unnecessary dependencies in `requirements.txt`.

[24/04/12] Add LoRA module detection. Make BlockOptimizer compatible with lr scheduler.