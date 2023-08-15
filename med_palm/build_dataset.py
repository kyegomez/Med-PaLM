import argparse
import multiprocessing
from itertools import chain

import torch
from datasets import load_dataset
from model import PALMETokenizer


class CFG:
    SEED: int = 42
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    HF_ACCOUNT_REPO: str = "YOUR HF ACCOUNT"
    TOKENIZER: str = "EleutherAI/gpt-neox-20b"
    DATASET_NAME: str = "EleutherAI/the_pile_deduplicated"



def prep_sample(sample):
    question = sample["question"]
    multiple_choice_answer = sample["multiple_choice_answer"]
    answers = sample["answers"]
    image_id = sample["image_id"]
    answer_type = sample["answer_type"]
    question_id = sample["question_id"]
    image = sample["image"]

    text = f"Question: {question} Multiple Choice Answer: {multiple_choice_answer} Answers: {answers} Answer Type: {answer_type} Question ID: {question_id} Image ID: {image_id}"
    
    return {
        "image": image,
        "target_text": text
    }



def main(args):
    tokenizer = PALMETokenizer()

    train_dataset = load_dataset(CFG.DATASET_NAME, split="train", streaming=True)

    def prep_and_group_texts(samples):
        processed_samples = []
        for sample in samples:
            prepared_sample = prep_sample(sample)
            text = prepared_sample["target_text"]
            image = prepared_sample["image"]

            text_tokens, _ = tokenizer.tokenize_texts([text + tokenizer.eos_token])
            image_tokens = tokenizer.tokenize_images([image])

            # Since both text and image tokens are tensors, concatenate them along the sequence dimension.
            merged_tokens = torch.cat((text_tokens, image_tokens), dim=-1)

            processed_samples.append(merged_tokens)

        # Concatenate all sequences.
        concatenated_examples = list(chain(*processed_samples))

        total_length = len(concatenated_examples)
        if total_length >= CFG.SEQ_LEN:
            total_length = (total_length // CFG.SEQ_LEN) * CFG.SEQ_LEN
        
        # Split by chunks of block_size.
        result = [t[i : i + CFG.SEQ_LEN] for i in range(0, total_length, CFG.SEQ_LEN)]
        return result

    train_tokenized_dataset = train_dataset.map(
        prep_and_group_texts,
        batched=True,
        # num_proc=CFG.NUM_CPU,
    )
    

    train_tokenized_dataset.push_to_hub(CFG.HF_ACCOUNT_REPO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=CFG.SEED, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=CFG.SEQ_LEN, help="Sequence length for processing")
    parser.add_argument("--hf_account", type=str, default=CFG.HF_ACCOUNT_REPO, help="Hugging Face account name and repo")
    parser.add_argument("--tokenizer", type=str, default=CFG.TOKENIZER, help="Tokenizer model to use")
    parser.add_argument("--dataset_name", type=str, default=CFG.DATASET_NAME, help="Name of the dataset to process")
    args = parser.parse_args()
    main(args)