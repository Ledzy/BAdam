import ast
import torch
import openai
import time
import re
import fastchat
from collections import defaultdict
from peft import PeftModel
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.tuner.core.trainer import PeftModelMixin
from trl.trainer.utils import pad_to_length

API_MAX_RETRY = 5
API_RETRY_SLEEP = 2
API_ERROR_OUTPUT = "$ERROR$"

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
max_length = 512

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from llmtuner.hparams import FinetuningArguments


class DPOPeftTrainer(PeftModelMixin, DPOTrainer):
    
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        **kwargs
    ):
        self.finetuning_args = finetuning_args
        self.ref_model = ref_model
        self.use_dpo_data_collator = True # hack to avoid warning
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.beta = finetuning_args.dpo_beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._prepare_judge()
        self.eval_counter = 0
        self.exp_name = None
        self.out_dir = "outputs/logs"
        self.response_model = "gpt-4"

        Trainer.__init__(self, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def concatenated_forward(
        self,
        model: Optional[torch.nn.Module] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()}) # avoid error
        unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(self.model)

        if not torch.is_grad_enabled():
            unwrapped_model.gradient_checkpointing_disable()

        if model is None and isinstance(unwrapped_model, PeftModel): # peft model has no ref_model
            with unwrapped_model.disable_adapter():
                all_logits = self.model(
                    input_ids=batch_copied["input_ids"],
                    attention_mask=batch_copied["attention_mask"],
                    return_dict=True
                ).logits.to(torch.float32)
        else:
            all_logits = model(
                input_ids=batch_copied["input_ids"],
                attention_mask=batch_copied["attention_mask"],
                return_dict=True
            ).logits.to(torch.float32)

        if not torch.is_grad_enabled():
            unwrapped_model.gradient_checkpointing_enable()

        all_logps = self._get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def _get_judges(self, questions, answers, parallel=16):
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        judges = []
        parallel = min(parallel, len(questions))
        with ThreadPoolExecutor(parallel) as executor:
            for judge in tqdm(executor.map(self._get_rating, questions, answers)):
                judges.append(judge)
        return judges
    
    def log_outputs(self, metrics):
        import os, json
        if self.exp_name is None:
            from datetime import datetime
            self.exp_name = datetime.now().strftime("%m%d-%H%M%S")
        fname = os.path.join(self.out_dir, self.exp_name)
        
        hist = []
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                hist = json.load(f)
        import ipdb; ipdb.set_trace()
        for i in range(len(metrics["question"])):
            block = {}
            for k in metrics.keys():
                if isinstance(metrics[k], int) or isinstance(metrics[k], str):
                    block[k] = metrics[k]
                else:
                    block[k] = metrics[k][i]
                    
            hist.append(block)
        
        final = json.dumps(hist, indent=2)
        with open(fname, 'w') as f:
            f.write(final)

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor], ref_out=False) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""
            
        # Avoid repeat evaluation on prompt
        bz = len(batch["prompt_ids"]) // 2
        for k in ["prompt_ids", "prompt_attention_mask"]:
            batch[k] = batch[k][:bz]
            
        with torch.no_grad():
            unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(self.model)
                
            # print(f"rank: {self.accelerator.process_index}")
            # return ["q1", "q2"], ["a1", "a2"]

            import time; t0 = time.time()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                policy_output = unwrapped_model.generate(
                    input_ids = batch["prompt_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            t1 = time.time()
            print(f"rank: {self.accelerator.process_index}, bz: {bz}, time: {t1-t0}, avg_time: {(t1-t0)/bz}")
            questions = self.tokenizer.batch_decode(batch["prompt_ids"], skip_special_tokens=True)
            policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
            answers = [a[len(q):] for (a, q) in zip(policy_output_decoded, questions)]
            if not ref_out:
                return questions, answers
            if not torch.is_grad_enabled():
                unwrapped_model.gradient_checkpointing_disable()
            with unwrapped_model.disable_adapter():
                reference_output = self.ref_model.generate(
                    input_ids = batch["prompt_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            if not torch.is_grad_enabled():
                unwrapped_model.gradient_checkpointing_enable()

        reference_output = pad_to_length(reference_output, max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded
    
    def _prepare_judge(self, judge_file=None):
        from fastchat.llm_judge.common import load_judge_prompts
        if judge_file is None:
            judge_file = "data/judge_prompts.jsonl"
        self.judge_prompts = load_judge_prompts(judge_file)["single-v1"]
        
    def _get_rating(self, question, answer, temperature=0, max_tokens=2048):
        """Given the question and answer pair, get the rating from the model"""
        from fastchat.model.model_adapter import get_conversation_template
        from fastchat.llm_judge.common import chat_compeletion_openai
        user_prompt = self.judge_prompts["prompt_template"].format(
            question=question,
            answer=answer,
        )
        sys_prompt = self.judge_prompts["system_prompt"]
        conv = get_conversation_template(self.response_model)
        conv.set_system_message(sys_prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        
        # get the judment response from the GPT3.5/GPT4 model
        response = chat_compeletion_openai(self.response_model, conv, temperature=temperature, max_tokens=max_tokens)
        match = re.search(one_score_pattern, response)
        if not match:
            match = re.search(one_score_pattern_backup, response)
        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
            print("cannot find rating from GPT4!")
        return rating, user_prompt, response