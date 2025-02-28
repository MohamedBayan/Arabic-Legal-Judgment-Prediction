from transformers import pipeline
import re
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Define model path as an environment variable or relative path
MODEL_PATH = os.getenv('MODEL_PATH', './base_models/Meta-Llama-3.1-8B-Instruct/')  # Default to local path if not set

import os
import json

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified path.
    """
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_text(model, tokenizer, ground_truth, model_response):
    sys_prompt = """
    You are an expert legal analyst and computer scientist specializing in evaluating the performance of language models in legal contexts. Your role is to critically assess the quality of a model’s response to a legal query by comparing it against a provided ground truth. Use your expertise in legal reasoning, language modeling, and formal legal writing to assign precise scores based on detailed evaluation criteria.
    """
    prompt = """
        #### Provided Information:
        - **Ground Truth:** 
        {ground_truth}

        - **Model Response:**
        {model_response}

        #### Evaluation Instructions:
        Evaluate the model's response using the following metrics, assigning scores on a scale of 1 to 10:
        1. **Accuracy:** Does the response align with the factual and legal details in the ground truth? (Score: 1-10)
        2. **Relevance:** Does the response directly address the legal question or matter at hand? (Score: 1-10)
        3. **Coherence:** Is the response logically organized and consistent? (Score: 1-10)
        4. **Brevity:** Is the response concise while maintaining sufficient detail? (Score: 1-10)
        5. **Legal Language:** Does the response use formal and accurate legal terminology? (Score: 1-10)
        6. **Faithfulness to Ground Truth:** Does the response preserve the facts and principles provided in the ground truth? (Score: 1-10)
        7. **Clarity:** Is the response clear and easy to understand? (Score: 1-10)
        8. **Consistency:** Does the response avoid contradictions and remain logically consistent? (Score: 1-10)

        #### Output Format:
        Return your evaluation as a JSON object using the following format:
        {{
        "Accuracy": <score>,
        "Relevance": <score>,
        "Coherence": <score>,
        "Brevity": <score>,
        "Legal Language": <score>,
        "Faithfulness to Ground Truth": <score>,
        "Clarity": <score>,
        "Consistency": <score>
        }}
        #### Important Note:
        Only return the JSON object. Do not provide any explanation or additional text.
        """
    prompt = prompt.format(ground_truth=ground_truth, model_response=model_response)

    messages = [
        {"role": "system", "content":sys_prompt},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        #add_generation_prompt=False,
        #continue_final_message=True,
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
    print(response)
    return response

# Read and save JSON functions with relative paths
def read_json(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except UnicodeDecodeError as e:
        print(f"Error decoding the file {file_path}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return None

def save_json(file_path, data, encoding='utf-8'):
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"File saved: {file_path}")
    except UnicodeEncodeError as e:
        print(f"Error encoding the file {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error saving {file_path}: {e}")

# Directory path as an environment variable or relative path
directory_path = os.getenv('RESULTS_PATH', './results/llama-3.2-ft')  # Default to local path if not set
i = 0
model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if os.path.isfile(file_path) and filename.endswith('.json'):
        print(f"Processing file: {filename}")
        
        data = read_json(file_path)
        if data is None:
            continue  
        
        ground_truth = data.get("output", "")
        model_response = data.get("response", "")
        
        if not ground_truth or not model_response:
            print(f"Missing data in {filename}, skipping.")
            continue
        response = generate_text(model, tokenizer, ground_truth=ground_truth, model_response=model_response)
        generated_text = response.strip()
        i += 1
        print("\n" * 5, i, generated_text)
        data["LLM-raw-evaluation"] = response
        
        match = re.search(r'{.*}', generated_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result_dict = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from string: {json_str[:100]}...") 
                print(f"Error message: {e}")
                result_dict = {}  
            
            # Save the result into the data
            data["LLM-evaluation"] = result_dict
            print("Model response:", result_dict)

            save_json(file_path, data)
        else:
            print(f"Error: No valid JSON found in the response of file {filename}")
