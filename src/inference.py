import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified path.
    """
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_text(model, tokenizer, instruction, input_text):
    """
    Generate text based on the given instruction and input text.
    """
    messages = [
        {"role": "system", "content": "أنت محامي قانوني. بناءً على الوقائع والأسباب المقدمة، حدد واكتب نص الحكم القانوني المناسب بدقة، مع الالتزام بالمبادئ القانونية المتوقعة."},
        {"role": "user", "content": f"{instruction}\n{input_text}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.001
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    response = response.split("assistant\n")[-1].strip()
    return response

def main(model_path, data_file, result_path):
    """
    Main function to process the dataset and generate responses.
    """
    os.makedirs(result_path, exist_ok=True)

    # Load dataset
    df = pd.read_json(data_file, lines=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Process each row in the dataset
    for _, row in df.iterrows():
        sample = row.to_dict()
        file_path = os.path.join(result_path, sample["id"] + ".json")

        # Skip if result file already exists
        if os.path.exists(file_path):
            continue

        # Remove embedding field
        del sample['embedding']

        # Generate response
        result = generate_text(model, tokenizer, sample["Instruction"], sample["input"])

        # Add response to sample
        sample["response"] = result

        # Save result to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Setup CLI argument parsing
    parser = argparse.ArgumentParser(description="Generate legal responses based on input dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the input dataset (JSONL format)")
    parser.add_argument('--result_path', type=str, required=True, help="Directory to save the generated results")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with CLI arguments
    main(args.model_path, args.data_file, args.result_path)
