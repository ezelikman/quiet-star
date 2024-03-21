import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral import configuration_mistral as original_configuration_mistral
from transformers.models.mistral import modeling_mistral as original_modeling_mistral

import configuration_mistral
import modeling_mistral

original_modeling_mistral.MistralModel = modeling_mistral.MistralModel
original_modeling_mistral.MistralForCausalLM = modeling_mistral.MistralForCausalLM
original_configuration_mistral.MistralConfig = configuration_mistral.MistralConfig

model_path = "ezelikman/quietstar-8-ahead"

n_ahead = 8
n_ahead_talk = 1
merged_talk_heads = True

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_8bit=True,
                                             max_thoughts=n_ahead + n_ahead_talk + 1,
                                             merged_talk_heads=merged_talk_heads,
                                             merged_lm_and_talk_heads=False,
                                             merged_lm_and_think_heads=True,
                                             use_concat_talk_head=True,
                                             use_shallow_think=True,
                                             use_shallow_talk=False,
                                             use_complex_think_head=False,
                                             use_complex_talk_head=True,
                                             use_weighted_talk_head=True,
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.use_end_thought_token = True
model.tokenizer = tokenizer
model.use_start_thought_token = True
model.wandb_enabled = True
model.n_ahead = n_ahead
model.n_passes = 1
model.eval_mode = True
model.first_run = False
model.kill_after = 100
model.rm_initialized = True

model.original_mode = False

input = "Solve the equation 2x + 3x² = 5."

input_ids = tokenizer.encode(input, return_tensors="pt").to(model.device)

# output = model.generate(input_ids, max_length=50)


def generate(input_ids, attention_mask, model, temp, max_length=20):
    with torch.no_grad():
        finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        for cur_token_idx in range(max_length):
            # Sample the next token
            new_ids = model(
                input_ids[~finished_generating],
                attention_mask=attention_mask[~finished_generating]
            )['logits']
            # Mask out the start and end thought tokens so we don't accidentally sample them
            new_ids[:, :, model.tokenizer.vocab_size:] = -float("inf")
            for list_idx, answer_idx in enumerate((~finished_generating).nonzero(as_tuple=True)[0]):
                # Find the index of the last token that is not padding
                base_answer_ids = input_ids[answer_idx]
                new_answer_ids = new_ids[list_idx]
                last_token_idx = (base_answer_ids != model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0].max()


                new_ids_sampled = torch.multinomial(
                        torch.nn.functional.softmax(new_answer_ids[last_token_idx] / temp, dim=-1), 1)
                # Assign the new id to the last token
                if last_token_idx + 1 >= len(base_answer_ids):
                    # Add padding everywhere
                    new_padding = torch.full((len(input_ids), 1), model.tokenizer.pad_token_id, dtype=torch.long,
                                             device=input_ids.device)
                    input_ids = torch.cat([input_ids, new_padding], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.zeros_like(new_padding)], dim=-1)
                attention_mask[answer_idx, last_token_idx + 1] = 1
                input_ids[answer_idx, last_token_idx + 1] = new_ids_sampled
                if new_ids_sampled == model.tokenizer.eos_token_id or new_ids_sampled == model.tokenizer.bos_token_id or new_ids_sampled == model.tokenizer.pad_token_id:
                    finished_generating[answer_idx] = 1
            if finished_generating.all():
                break
    return input_ids, attention_mask


out = generate(input_ids, torch.ones_like(input_ids), model, 0.9, max_length=100)


print(tokenizer.decode(out[0][0], skip_special_tokens=False))

print("hi")