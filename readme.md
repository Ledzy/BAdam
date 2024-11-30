# BAdam

The implementation for [BAdam: A Memory Efficient Full Parameter Optimization Method for Large Language Models](https://arxiv.org/abs/2404.02827). This paper presents an algorithm named **BAdam**, which finetunes Llama 2-7b and Llama 3-8B using **a single RTX3090** with Adam's update rule and mixed precision training. The core idea of **BAdam** is to sequentially solve block coordinate optimization sub-problems. From the implementation perspective, the algorithm runs Adam's update on a small portition (usually one single transformer layer) of the parameters, thereby requires much less memory in comparison to full parameter Adam finetuning. **Using BAdam only requires one line modification of the original code.**

| Method | Minimum Memory | Actual Memory Cost (Llama 3-8B) | Actual Memory Cost (Llama 2-7B) |
| -------- | -------- | -------- | -------- |
| Adam    | $18M$     | 144 GB+ | 122.8 GB+     |
| **BAdam**    | $2M + \frac{16M}{D}$   | 23.5 GB|  21.8 GB     |
<!-- | LoRA    | Data     | Data     | -->
**Table 1: Comparison of Methods.** $M$ stands for the number of model's parameters in billion and $D$ is the number of blocks used in **BAdam**. See Table 2 in paper for detailed analysis on memory consumption.

| Method | Llama 3-8b | Llama 2-7b |
| -------- | -------- | -------- | 
| Pretrained model | 5.46 | 3.93 |
| LoRA | 6.41   | 4.86 | 
|  **BAdam**  | **6.67** | **5.21** |
<!-- | LoRA    | Data     | Data     | -->
**Table 2: MT bench score.** The model is instruction finetuned on Alpaca-GPT4 dataset using a single RTX3090. **BAdam** consistently outperforms LoRA in MT bench under various evaluation models.

One can also apply **BAdam** for larger models with size such as 13B, 22B, 30B, and 70B. The memory consumption can be estimated to be $2M + \frac{16M}{D}$ (GB), plus some additional memory consumption for gradient checkpointed activations and system use like PyTorch's pre-allocation, etc (minor part). When using model parallelism with $N$ GPUs, the memory cost can be estimated by $\frac{2M + 16M/D}{N}$ (GB), plus the additional communication buffers.

## Change log
[24/09/26] **BAdam** has been accepted by NeurIPS, 2024!

[24/06/16] We support model parallel using Deepspeed ZeRO-3 now!

[24/04/16] Our algorithm has been added to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We would like to express our gratitude to their efforts on integrating **BAdam**!

[24/04/12] Add LoRA module detection. Make BlockOptimizer compatible with lr scheduler.

## Table of Contents
- [Environment Setup](#setup)
- [Usage of BAdam](#usage-of-badam)
    - [Partition by Module (A Single GPU)](#partition-by-module-a-single-gpu)
    - [Partition by Module (Model Parallel)](#partition-by-module-model-parallel)
    - [Partition by Parameter Ratio](#partition-by-parameter-ratio)
    - [Hyperparameter Suggestion](#hyperparameter-suggestion)
- [Run Paper Experiment](#run-paper-experiment)

## Setup
To install **BAdam** from Pypi, one can run:
```bash
pip install badam
```

One may also choose to build from source by the following steps:
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

### Partition by Module (A Single GPU)
**BAdam** uses mixed-precision training, make sure that the model is loaded in `float16` precision for memory saving. One can simply add **one line of code** that wraps the original optimizer to use **BAdam**.

```python
from badam import BlockOptimizer

# before training, add this line to wrap the original optimizer
optimizer = BlockOptimizer(
    base_optimizer=original_optimizer, # can be any torch.Optimizer
    named_parameters_list=list(model.named_parameters()), 
    switch_block_every=100, # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter. 
    switch_mode="random", # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
    verbose=2 # information level, will print trainable parameters when setting to 2
)
```
The above code automatically creates a block partition according to `model.named_parameters`. Specifically, it treates each transformer layer module as a single block. For instance, for Llama 3-8B, the block partition ($D = 32$) will be
```
block 1: model.layers.0.
block 2: model.layers.1.
...
block 32: model.layers.31.
```
**By default, the embedding layer and language modeling head is not included in the training blocks**. One can add them as two additional blocks by setting `include_embedding=True`, `include_lm_head=True`. 

<details><summary>Click to see more partition strategies and example code</summary>

One can also specify their own block list for the block optimizer. This can be achieved by adjusting the `block_prefix_list` argument. For instance, the following code snippets creat block partitions by self_attn and mlp modules (i.e., D = 32 * 2 = 64 for Llama 3-8B), and matrix modules (i.e., D = 32 * 7=224 for Llama 3-8B), respectively, which helps further reduce the memory cost:

```python
# block partition by self_attn and mlp modules
block_prefix_list = []
for i in range(32):
    layer_prefix = [
        [f"model.layers.{i}.self_attn."],
        [f"model.layers.{i}.mlp."],
    ]
    block_prefix_list.extend(layer_prefix)

optimizer = BlockOptimizer(
    base_optimizer=original_optimizer,
    named_parameters_list=list(model.named_parameters()), 
    switch_block_every=100,
    switch_mode="random",
    verbose=2,
    block_prefix_list=block_prefix_list # set the block list
)
```

```python
#block partition by matrix modules
block_prefix_list = []
for i in range(32):
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
    switch_block_every=100,
    switch_mode="random",
    verbose=2,
    block_prefix_list=block_prefix_list # set the block list
)
```
We have tested that block partition by self_attn and mlp modules achieves a MT-bench score 6.65 for finetuning Llama 3-8B. This score matches that (6.67) achieved by block partition by transformer layer modules, while further reduces the memory cost. 

</details>

**Important Notes:**
* When setting block partition, one should be careful with the downstream task. Some tasks has randomly initialized classification layers, such as the SuperGLUE where the `task_dict` and `pooler` layers are _randomly initialized_. **In this case, make sure to train these layers first, or set it to be trainable through the whole time.** To set modules to be trainable through the whole training process, one can use `active_modules` argument, e.g., set `active_modules=["model.task_dict.", "model.pooler."]` when create the BlockOptimizer. Note that randomly initialized layers are usually the last layer, so updating these layers will only introduce negligible additional BP time. We thus suggest to always set the last classification layer to be trainable when the memory is permitted, if it is randomly initialized.
* The parameters that are not included in `block_prefix_list` will be inactive (freezed) through the whole training procedure.
* When setting prefix, it is suggested to include a `.` at the end. For example, it is preferred to use `model.layers.1.` instead of `model.layers.1`, as the later one includes the layer 10, 11, ..., 19 as well (since they have the same prefix).

### Partition by Module (Model Parallel)
We support the model parallel offered by deepspeed ZeRO-3. It partitions the model, gradient, and optimizer states across different GPUs so that one can train large models (e.g., Llama 3-70B) that cannot be fit into a single GPU.  Given $N$ GPUs, the per GPU memory cost can be estimated by $\frac{2M + 16M/D}{N}$ (GB), plus the additional cost for communication buffer and temporary parameter gathering buffer arised during forward/backward. These buffer sizes can be configurated manually and determines the efficieny of the communication system.

<details><summary>Click to see instructions for model parallelism</summary>

To use ZeRO-3, one needs to set `ds_zero3_enabled=True` when initializing the BlockOptimizer. Then, set `block_optimizer.ds_optimizer = ds_optimizer` after calling `deepspeed.initialize`. 

```python
from badam import BlockOptimizer

optimizer = BlockOptimizer(
    ...,
    ds_zero3_enabled=True # set it to True
)

model, ds_optimizer = deepspeed.initialize(model=model, optimizer=optimizer, ...)

# create the reference to the ds_optimizer, for the purpose of setup ZeRO-3's environment
optimizer.ds_optimizer = ds_optimizer
```

When using huggingface Trainer to control the workflow, accessing ds_optimizer is not direct. One can add the BAdamCallback which automatically handles the reference to ds_optimizer:

```python
from badam.utils import BAdamCallback

callbacks = original_callbacks.append(BAdamCallback) # add the callback
trainer = YourTrainerClass(
    ...,
    callbacks=callbacks
)
```
The model parallelism results in noticable overhead due to the communication cost. In particular, we empirically observe about 3 times overhead when training Llama 3-8B with 4 RTX3090 GPUs (without NVLink) using ZeRO-3, in comparison to using a single GPU, under the same `per_device_batch_size`. Fortunately, one may use a larger `per_device_batch_size` to accelerate the training speed as ZeRO-3 greatly reduces the per GPU memory cost.

Make sure to use `accelerate config` to configurate the distributed training and then use proper command to launch your script in a distributed way, such as `accelerate launch` and `deepspeed`.

</details>



### Partition by Parameter Ratio
Instead of partitioning block by the model's parameter, an alternative choice is to train all the parameters simultaneously with a fixed ratio. For instance, we can train 5% parameters of every transformer layer. Namely, each active block contains 5% parameters from every transformer layer. In this sense, the feature extractor of every layer are jointly trained, which may be preferred in certain scenarios. However, training a block consisting of parameters coming from all the transformer layers may lose partly the benefit of BP time saving of **BAdam**.


<details><summary>Click to see example code and instructions</summary>

```python
from badam import BlockOptimizerRatio

optimizer = BlockOptimizerRatio(
    param_groups=param_groups, # param_group of torch.Optimizer, the same as the original optimizer
    named_parameters_list=list(self.model.named_parameters()),
    switch_every=100, # switch to the new block every 100 updates
    update_ratio=0.1, # ratio of trainable weight for each parameter
    mask_mode = "adjacent", # choices: ["adjacent", "scatter"], see Note below for more explanation
    lr=1e-6,
    betas=(0.9, 0.999), # betas for Adam update
    eps=1e-8, # eps of Adam update
)
```
Currently, the `BlockOptimizerRatio` only supports the `Adam` update. The repository is still under active development.

**Notes:**
* The `mask_mode` indicates how should the trainable parameter distribute across a parameter. `mask_mode=adjacent` indicates that the trainable parameters are adjacent to each other, while `mask_mode=scatter` indicates that trainable parameters are randomly choosed from the weight. For instance, considering optimizing a $10 \times 10$ matrix with `update_ratio=0.1`, setting `mask_mode=adjacent` will let parameters of the same row be the same block, and `mask_mode=scatter` means randomly choose 10 trainable parameters from the matrix.
* By default, `BlockOptimizerRatio` does not update embedding layer, since in principle the embedding vectors of the tokens that are included in the training samples should be updated, while randomly freeze embedding parameters makes the update imbalanced. One can set `include_embedding=True` to include it for experimental purpose.
* For `BlockOptimizerRatio`, we notice that setting `mask_mode = "adjacent"` usually performs the best; we leave the study of `mask_mode` as a future work. The convergence speed is highly positively related to the `update_ratio`, so we suggest to choose it as high as possible when the memory is permitted. 
* The gradient and optimizer states are stored in sparse tensor format. The update rule is exactly the same as the  `BlockOptimizer`: run Adam update on current active block for `switch_every` steps, and then switch to next block.
* Currently, the operation of sparsifing the gradient causes noticable overhead, which inevitably slow down the training. We leave the acceleration as a future work.
</details>

### Hyperparameter Suggestion
* Choice of the `switch_block_every`. Compared to Adam, our BAdam only introduces _one_ additional hyperparameter, i.e., the `switch_block_every` (the `K` Adam steps in paper). It determines how many Adam steps we perform for each active block before switching to the next one. Fortunately, this hyperparameter can be set adaptively. Ideally, we expect to balance the data usage for each block in every epoch. This gives a natural choice of `switch_block_every` = $\frac{n}{BD}$ (rounding to the nearest integer if it is a fractional), where $n$ is the number of training data points, $B$ is the effective batch size, and $D$ is the number of blocks in BAdam. Using such a setting ensures that after one block-epoch, all the training data points are equally distributed to the $D$ blocks for training. Meanwhile, to achieve sufficient decrease for each block coordinate descent subproblem and fully utilize the advantage of mixed precision training for reducing rounding error, the switch frequency should not be too small. Additionally, too large switch frequency may over-optimize one block before moving to others. **We notice that setting** `switch_block_every`  = $\min(\max(\frac{n}{BD}, 50),100)$ **usually yields fast convergence speed on both training loss and validation loss.**


## Run Paper Experiment

<details><summary>Llama 3-8B and Llama 2-7B on Alpaca-GPT4</summary>

Our implementation of finetuning Llama 3 and Llama 2 is based on [Llama Factory](https://github.com/hiyouga/LLaMA-Factory). This repository mainly serves as the purpose for reproducing our paper's results. For better support on advanced algorithmic features, we suggest to use the latest version of Llama Factory. 

For the experiment of finetuning Llama-2 7b on [Alpaca-GPT4](https://arxiv.org/abs/2304.03277) dataset, change the working directory to `llama`:
```bash
cd llama-alpaca
```
Here is a sample command for running the code:
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type block \
    --output_dir ./outputs/llama2-7b \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --val_size 500 \
    --eval_steps 20 \
    --evaluation_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 100 \
    --switch_mode random \
    --bf16 True
```
To finetune Llama 3-8B, one can set `--model_name_or_path meta-llama/Meta-Llama-3-8B`. We use learning rate `1e-6` for Llama 3-8B and learning rate 1e-5 for Llama 2-7B, respectively. It is important to note that the favorable learning rate may vary for different models and datasets. 

**Notes on arguments:**
* `--stage`: Currently we only implement the `sft`.
* `--finetuning_type`: Options: (block, full, lora, sparse)
* `--switch_mode`: How to order the block update. Options: (random, ascending, descending).
* `--switch_block_every`: Switch block frequency; see "Hyperparameter Suggestion" for how to set this hyperparamter.
* The above sample command is different from the hyperparameters settings in paper, while this version is more efficient. We will update our paper later. 
</details>

<details><summary>RoBERTa-large on SuperGLUE</summary>

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
</details>
