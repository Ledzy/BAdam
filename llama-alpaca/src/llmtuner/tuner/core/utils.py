import torch
import re

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

llama_param_groups = ['model.embed_tokens', 'model.layers.0', 'model.layers.1', 'model.layers.2', 'model.layers.3', 
                      'model.layers.4', 'model.layers.5', 'model.layers.6', 'model.layers.7', 'model.layers.8', 
                      'model.layers.9', 'model.layers.10', 'model.layers.11', 'model.layers.12', 'model.layers.13', 
                      'model.layers.14', 'model.layers.15', 'model.layers.16', 'model.layers.17', 'model.layers.18', 
                      'model.layers.19', 'model.layers.20', 'model.layers.21', 'model.layers.22', 'model.layers.23', 
                      'model.layers.24', 'model.layers.25', 'model.layers.26', 'model.layers.27', 'model.layers.28', 
                      'model.layers.29', 'model.layers.30', 'model.layers.31']

# llama_param_groups.extend(['model.norm', 'lm_head'])

def get_rating(judge_prompts, question, answer, temperature=0, max_tokens=2048, response_model="gpt-4"):
    """Given the question and answer pair, get the rating from the model"""
    from fastchat.model.model_adapter import get_conversation_template
    from fastchat.llm_judge.common import chat_compeletion_openai
    user_prompt = judge_prompts["prompt_template"].format(
        question=question,
        answer=answer,
    )
    sys_prompt = judge_prompts["system_prompt"]
    conv = get_conversation_template(response_model)
    conv.set_system_message(sys_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    
    # get the judment response from the GPT3.5/GPT4 model
    response = chat_compeletion_openai(response_model, conv, temperature=temperature, max_tokens=max_tokens)
    match = re.search(one_score_pattern, response)
    if not match:
        match = re.search(one_score_pattern_backup, response)
    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
        print("cannot find rating from GPT4!")
    return rating, user_prompt, response

def prepare_judge(judge_file=None):
    from fastchat.llm_judge.common import load_judge_prompts
    if judge_file is None:
        judge_file = "data/judge_prompts.jsonl"
    return load_judge_prompts(judge_file)["single-v1"]

def count_param(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if any([p.requires_grad, not trainable]))

def bp():
    from accelerate import Accelerator
    accelerator = Accelerator()
    if accelerator.is_main_process:
        import ipdb; ipdb.set_trace()
    accelerator.wait_for_everyone()

def set_trainable_block(model, optimizer, block_idx, verbose=2):
    """quick function for adjust trainable parameter. To be integrated to optimizer later"""
    trainable_param = llama_param_groups[block_idx]
    if verbose >= 1:
        print(f"trainable block:{trainable_param}")

    # Freeze the param if not in trainable params
    for name, param in model.named_parameters():
        # if not any(p in name for p in trainable_params):
        if not trainable_param in name:
            param.requires_grad_(False)
            param.grad = None
        else:
            if verbose >= 2:
                print(name)
            param.requires_grad_(True)
            # param.data = param.data.to(torch.float32)
            
    # Clean optimizer state, e.g. momentum term and Adam's exp_avg and exp_avg_sq term
    if optimizer is not None:
        from collections import defaultdict
        for group in optimizer.param_groups:
            for p in group['params']:
                # unwrap optimizer to make sure state attribute exists
                while hasattr(optimizer, "optimizer"):
                    optimizer = optimizer.optimizer

                # from accelerate import Accelerator
                # accelerator = Accelerator()
                # if accelerator.is_main_process:
                #     import ipdb; ipdb.set_trace()
                # accelerator.wait_for_everyone()
                
                optimizer.state[p] = defaultdict()
    
    return model

def clean_grad_hook(model: torch.nn.Module):
    """hook for cleaning the intermediate gradients when backward through the network"""
    def hook(x):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad = None
    
    return hook

def set_trainable(model, param_map=None, rank=None, optimizer=None, block_idx=None, memory_limit=5, num_gpus=4, verbose=1, float32=False):
    """Set trainable params according to memory budget, For LlaMa 2 usage only
    Args:
        model (nn.Module): Should be the pretrained model, i.e. without value head wrap
        rank (int): rank of the process
        param_map (Dict): {"layer_name": device_id}
        memory_limit (list or integer): specify the memory limit for each GPU
        num_gpus (int)
        float32 (bool): whether using float32 to store the model.
    """
    if type(memory_limit) == float or type(memory_limit) == int:
        memory_limit = [memory_limit * 1024 ** 3] * num_gpus # trasfer to Gigabytes
    elif type(memory_limit) == list:
        for i in range(len(memory_limit)):
            memory_limit[i] *= 1024 ** 3
    
    bit_per_param = 4
    if not float32: 
        # i.e. use float16
        bit_per_param = 2
        
    if param_map is None:
        assert rank is not None # need rank to control memory allocation
        # the whole model is on the same device
        param_map = {param: rank for param in llama_param_groups}
    
    # from accelerate import Accelerator
    # accelerator = Accelerator()
    # if accelerator.is_main_process:
    #     import ipdb; ipdb.set_trace()
    # accelerator.wait_for_everyone()
    
    trainable_params = []
    
    if block_idx is not None:
        assert rank is not None # TODO: to be adapted for model parallelism
        trainable_params = ['lm_head', 'model.norm']
        if block_idx >= 0:
            trainable_params.append(f"model.layers.{block_idx}")
        
    else:
        for i in range(num_gpus):
            local_params = [p for p in param_map.keys() if param_map[p] == i]
            local_mem_used = 0

            for j in torch.randperm(len(local_params)):
                p_name = local_params[j]
                if 'layers' in p_name:
                    incom_param_count = count_param(getattr(model.base_model, 'layers')[int(p_name.split(".")[-1])])
                elif 'embed_tokens' in p_name:
                    incom_param_count = count_param(getattr(model.base_model, 'embed_tokens'))
                elif 'norm' in p_name:
                    incom_param_count = count_param(getattr(model.base_model, 'norm'))
                elif 'lm_head' in p_name:
                    incom_param_count = count_param(getattr(model, 'lm_head'))
                else:
                    incom_param_count = 0 # for non-llama model
                
                if local_mem_used + incom_param_count * bit_per_param < memory_limit[i]:
                    trainable_params.append(p_name)
                    local_mem_used += incom_param_count * bit_per_param
    
    if verbose >= 1:
        print("trainable parameters are", trainable_params)
        
    # Freeze the param if not in trainable params
    for name, param in model.named_parameters():
        if not any(p in name for p in trainable_params):
            param.requires_grad_(False)
            param.grad = None
            
            # handle = param.register_hook(clean_grad_hook(model)) # TODO: clean the last no grad block's gradient
        else:
            if verbose >= 2:
                print(name)
            param.requires_grad_(True)
            # param.grad = torch.zeros_like(param) # for handling with DDP reduce issue
            # param.data = param.data.to(torch.float32)
            
    # Clean all optimizer state, e.g. momentum term and Adam's exp_avg term
    # TODO: implement cpu offload
    if optimizer is not None:
        from collections import defaultdict
        # unwrap optimizer to make sure state attribute exists
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        for group in optimizer.param_groups:
            for p in group['params']:
                # if p.grad is not None:
                optimizer.state[p] = defaultdict()
    
    import gc; gc.collect()
    
    return model


opt = torch.optim.AdamW