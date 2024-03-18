import torch
import random
from transformers import AutoTokenizer

initial_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
initial_tokenizer.padding_side = "right"
initial_tokenizer.pad_token_id = initial_tokenizer.eos_token_id
eval_answer_marker="\nA:"

def preprocess_function(examples):
    dataset_transform = lambda xs: xs["text"]
    all_tokenized = [initial_tokenizer.encode(t, return_tensors="pt") for t in dataset_transform(examples)]
    new_tokenized = [{"input_ids": t} for t in all_tokenized]
    for i, t in enumerate(new_tokenized):
        new_tokenized[i]["input_ids"] = truncate_or_pad(t['input_ids'], initial_tokenizer.pad_token_id)
    new_input_ids = torch.cat([t["input_ids"] for t in new_tokenized], dim=0)
    new_attention_mask = (new_input_ids != initial_tokenizer.pad_token_id).long()
    tokenized = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def preprocess_eval_function_gsm(examples, use_few_shot=False, max_length=256):
    to_answer = lambda q, a: "Q: " + q + eval_answer_marker + a.split("####")[-1] + "\n"
    all_prompts = [to_answer(q, a) for q, a in zip(examples['question'], examples['answer'])]
    all_tokenized = [initial_tokenizer.encode(p, return_tensors="pt") for p in all_prompts]
    new_tokenized = [{"input_ids": t} for t in all_tokenized]
    for i, t in enumerate(new_tokenized):
        new_tokenized[i]["input_ids"] = truncate_or_pad(t['input_ids'], initial_tokenizer.pad_token_id, max_length)
    new_input_ids = torch.cat([t["input_ids"] for t in new_tokenized], dim=0)
    new_attention_mask = (new_input_ids != initial_tokenizer.pad_token_id).long()
    tokenized = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def preprocess_eval_function_csqa(examples, max_length=256):
    def construct_question(q, choices):
        choice_list = "\n".join([f"({label}) {choice}" for label, choice in zip(choices["label"], choices["text"])])
        return f"Q: {q}" + "\n" + choice_list
    to_answer = lambda q, c, a: construct_question(q, c) + eval_answer_marker + " " + a + "\n"
    all_prompts = [to_answer(q, c, a) for q, c, a in zip(examples['question'], examples['choices'], examples['answerKey'])]
    all_tokenized = [initial_tokenizer.encode(p, return_tensors="pt") for p in all_prompts]
    new_tokenized = [{"input_ids": t} for t in all_tokenized]
    for i, t in enumerate(new_tokenized):
        new_tokenized[i]["input_ids"] = truncate_or_pad(t['input_ids'], initial_tokenizer.pad_token_id, max_length)
    new_input_ids = torch.cat([t["input_ids"] for t in new_tokenized], dim=0)
    new_attention_mask = (new_input_ids != initial_tokenizer.pad_token_id).long()
    tokenized = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def compute_metrics(eval_pred, filter_numbers=True):
    logits, labels, _ = eval_pred
    accuracy = 0
    valid_number_tokens = [28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774, 28734, 13] # numbers
    valid_letter_tokens = [330, 365, 334, 384, 413, 13] # answer tokens
    for question, logits_guess in zip(labels, logits):
        # find which token corresponds to eval_answer_marker
        # chop off tokens from the end until the number of eval_answer_marker goes down
        detokenized_question = initial_tokenizer.decode(question)
        is_numeric = detokenized_question.split(eval_answer_marker)[-1][1].isdigit()
        valid_tokens = valid_number_tokens if is_numeric else valid_letter_tokens
        answer_count = detokenized_question.count(eval_answer_marker)
        for i in range(len(question) - 1, 0, -1):
            tokenized_subquestion = question[:i]
            if tokenized_subquestion[-1] == initial_tokenizer.pad_token_id:
                continue
            detokenized_subquestion = initial_tokenizer.decode(question[:i])
            if detokenized_subquestion.count(eval_answer_marker) < answer_count:
                break
        correct_answer_prob = 1
        # if is_numeric, then the first token just indicates that it's a number
        question_offset = 1 if is_numeric else 0
        for j in range(i + question_offset, len(question) - 1):
            if question[j + 1] == initial_tokenizer.pad_token_id:
                break
            true_token = question[j + 1]
            guess = torch.nn.functional.softmax(torch.tensor(logits_guess), dim=-1)
            # we only care about the logits assigned to the correct token
            if filter_numbers:
                if true_token not in valid_tokens:
                    continue
                guess_filtered = torch.zeros_like(guess)
                guess_filtered[:, valid_tokens] = guess[:, valid_tokens]
                guess_filtered = guess_filtered / guess_filtered.sum(dim=-1, keepdim=True)
                token_prob = guess_filtered[j, true_token]
            else:
                token_prob = guess[j, true_token]
            correct_answer_prob *= token_prob
        accuracy += correct_answer_prob / len(labels)
    return {"accuracy": accuracy}

def truncate_or_pad(t, padding_idx=0, max_length=256):
    if t.shape[1] > max_length:
        start = random.randint(0, t.shape[1] - max_length)
        t = t[:, start:start + max_length]
    else:
        padding = torch.zeros(t.shape[0], max_length - t.shape[1], dtype=t.dtype, device=t.device)
        t = torch.cat([t, padding + padding_idx], dim=1)
    return t
