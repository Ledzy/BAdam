# BAdam
The implementation for BAdam: A Memory Efficient Full Parameter Training Method for Large Language Models

## Setup
It is suggested to keep the following packages updated:
```
torch>=2.1.0
transformers>=4.35.2
```

## Quick Start
To use the **BAdam**, one can simply add one line of code that wraps the original optimizer.

```python
from block_optim import BlockOptimizer

# before training, add this line to wrap the original optimizer
optimizer = BlockOptimizer(
    base_optimizer=original_optimizer, # can be any torch.Optimizer
    named_parameters_list=list(model.named_parameters()), 
    switch_block_every=50, # switch to the new block every 50 updates
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
* The parameters that are not included in `block_prefix_list` will be freezed through the whole training procedure.
* When setting prefix, it is suggested to include a `.` at the end. For example, it is preferred to use `model.layers.1.` instead of `model.layers.1`, as the later one includes the layer 10, 11, ..., 19 as well (since they have the same prefix).

## Partition by parameter ratio
Instead of partitioning block by the model's parameter, an alternative choice is to train all the parameters with fixed ratio simultaneously. For instance, we can train $5\%$ of every parameter. In this sense, the feature extractor of every layer are jointly trained, which may be preferred in certain scenarios.

To do this, one can use the `SparseGradOptimizer`:

```python
# param_groups = optimizer.param_groups
from block_optim import SparseGradOptimizer

optimizer = SparseGradOptimizer(
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
Currently, the `SparseGradOptimizer` only supports the `Adam` update. The repository is still under active development.

**Note:**
* The `mask_mode` indicates how should the trainable parameter distribute across a parameter. `mask_mode=adjacent` indicates that the trainable parameters are adjacent to each other, while `mask_mode=scatter` indicates that trainable parameters are randomly choosed from the weight. For instance, for a $10 \times 10$ matrix, setting `mask_mode=adjacent` will let parameters of the same row be the same block, and `mask_mode=scatter` means randomly choose 10 trainable parameters from the matrix.
* The gradient and optimizer states are stored in sparse tensor format. The update rule is exactly the same as the  `BlockAdamOptimizer`: run Adam update on current active block for `switch_every` steps, and then switch to next block.
