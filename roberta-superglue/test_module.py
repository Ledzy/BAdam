import sys
from modulefinder import ModuleFinder

# badam_ft.py \
#     --task_name boolq \ # options: (boolq, wic, wsc, rte, multirc, copa)
#     --num_train_epochs 32 \
#     --eval_every_steps 100 \
#     --use_block_optim \ # delete this line to use full Adam, or change to `--use_sparse_optim` to use SparseGradOptimizer
#     --switch_every 100 \ # switch frequency
#     --switch_mode ascending \ # options: (ascending, descending, random) valid when `--use_block_optim` is set
#     --train_batch_size 16 \
#     --train_last_layer \ # whether to train the last layer. Set to true for superGLUE tasks as the last layer is randomly initialized
#     --hf_pretrained_model_name FacebookAI/roberta-large
sys.argv[1:] = [
    '--task_name', 'boolq',
    '--num_train_epochs', '32',
    '--eval_every_steps', '100',
    '--use_block_optim',
    '--switch_every', '100',
    '--switch_mode', 'ascending',
    '--train_batch_size', '16',
    '--train_last_layer',
    '--hf_pretrained_model_name', 'FacebookAI/roberta-large'
]

finder = ModuleFinder()
finder.run_script("badam_ft.py")

print('Loaded modules:')
for name, mod in finder.modules.items():
    print('%s: ' % name, end='')
    print(','.join(list(mod.globalnames.keys())[:3]))