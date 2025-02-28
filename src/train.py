import click
import torch
torch.autograd.set_detect_anomaly(True)

from datasets import load_from_disk
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)


@click.command()
@click.option('--model_name', default='model_name', help='Model name')
@click.option('--max_seq_length', default=512, help='Maximum sequence length')
@click.option('--quant_bits', default=4, help='Use 4-bit or 8-bit quantization')
@click.option('--use_nested_quant', default=False, help='Use double quantization')
@click.option('--max_steps', default=10, help='Maximum training steps')
@click.option('--batch_size', default=1, help='Batch size')
@click.option('--grad_size', default=2, help='Grad size')
@click.option('--epochs', default=1, help='number of epochs')
@click.option('--out_dir', default='outputs', help='Output directory')
@click.option('--save_steps', default=5, help='Save steps')
@click.option('--train_set_dir', default='/path/to/train_dataset', help='Training dataset directory')
@click.option('--dev_set_dir', default='/path/to/dev_dataset', help='Development dataset directory')
@click.option('--model_path', default='/path/to/model', help='Model path')
@click.option('--start_from_last_checkpoint', default=False, help='Start from last checkpoint')
@click.option('--lora_adapter_dir', default='lora_adapter', help='LoRA adapter directory')
@click.option('--merged_model_dir', default='merged_model', help='Merged model directory')
def main(model_name, max_seq_length, quant_bits, use_nested_quant, max_steps, batch_size, out_dir, save_steps,
         train_set_dir, dev_set_dir, model_path, start_from_last_checkpoint, grad_size, lora_adapter_dir,
         merged_model_dir, epochs):
    device_map = "DDP"  # Change as needed
    #device_map = "auto"

    if device_map == "DDP":
        device_string = PartialState().process_index
        device_map = {'': device_string}
        print(f"{device_map=}")

    HAS_BFLOAT16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if HAS_BFLOAT16 else torch.float16

    load_in_8bit = quant_bits == 8
    load_in_4bit = quant_bits == 4
    bnb_config = BitsAndBytesConfig(
    load_in_8bit=load_in_8bit,  # to enable 8-bit quantization rather than 4
    load_in_4bit=load_in_4bit,
    bnb_4bit_use_double_quant=use_nested_quant,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config= bnb_config if (load_in_4bit or load_in_8bit) else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_seq_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = max_seq_length
    tokenizer.padding_side = "right"
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if load_in_4bit or load_in_8bit:
        print("Loading model in " + str(dtype) + " bits")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)

    dataset = load_from_disk(train_set_dir)

    from multiprocessing import cpu_count
    def apply_chat_template(example, tokenizer):
        system_prompt = "أنت محامي قانوني. بناءً على الوقائع والأسباب المقدمة، حدد واكتب نص الحكم القانوني المناسب بدقة، مع الالتزام بالمبادئ القانونية المتوقعة."
        instruction = example["Instruction"]
        input = example["input"]
        output = example["output"]
        message = []
        message.append({"role":"system", "content": system_prompt})
        message.append({"role":"user", "content": instruction + "\n"+ input})
        message.append({"role":"assistant", "content":output})

        example["Text"] = tokenizer.apply_chat_template(message, tokenize=False, max_length=800, truncation=True)
        return example
    dataset = dataset.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                desc="Applying chat template",)
    print(dataset["Text"][0])

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="Text",
        max_seq_length=max_seq_length,
        packing=True,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_size,
            warmup_steps=10,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            fp16=not HAS_BFLOAT16,
            bf16=HAS_BFLOAT16,
            logging_steps=1,
            output_dir=out_dir,
            optim="adamw_hf",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="wandb",
            save_steps=save_steps,
            save_total_limit=3,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    logging.set_verbosity(logging.CRITICAL)

    trainer_stats = trainer.train(resume_from_checkpoint=start_from_last_checkpoint)

    print(f"{PartialState().process_index=}")
    trainer.model.save_pretrained(out_dir+f"/lora-epc{epochs}", save_adapter=True, save_config=True)

    trainer.model.save_pretrained(lora_adapter_dir, save_adapter=True, save_config=True)

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter_dir, torch_dtype=dtype, autocast_adapter_dtype=False)

    merged_model = model_to_merge.merge_and_unload()

    merged_model.save_pretrained(merged_model_dir)
    trainer.tokenizer.save_pretrained(merged_model_dir)

    file_path = 'trainer_stats.json'
    with open(file_path, 'w') as file:
        json.dump(trainer_stats, file, indent=4)
        print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")


if __name__ == "__main__":
    main()
