# [Arabic Legal Judgment Prediction (ArLJP)](<(https://arxiv.org/abs/2501.09768)>)

[![Paper](https://img.shields.io/badge/Paper-Download%20PDF-green)](https://arxiv.org/abs/2501.09768)

## **Overview**

This repository provides resources for **Arabic Legal Judgment Prediction (ArLJP)**, including datasets, fine-tuned models, and evaluation scripts. The research investigates the applicability of **Large Language Models (LLMs)** in predicting judicial outcomes for Arabic legal cases.

The study introduces:

- A **new Arabic instruction-following dataset for Legal Judgment Prediction (LJP)**
- Fine-tuned **LLaMA-3.2-3B** and **LLaMA-3.1-8B** models
- An **evaluation framework** combining BLEU, ROUGE, BERT, and qualitative assessments

## **Dataset**

**[Hugging Face: Arabic-LJP](https://huggingface.co/datasets/mbayan/Arabic-LJP)**  
Collected from the **Saudi Ministry of Justice**, this dataset includes court case **facts, legal reasoning, and judicial outcomes**, structured for LJP model training.

## **Models**

 **Fine-tuned Models:**

- [Llama-3.2-3B-ArLJP](https://huggingface.co/mbayan/Llama-3.2-3b-ArLJP)
- [Llama-3.1-8B-ArLJP](https://huggingface.co/mbayan/Llama-3.1-8b-ArLJP)

Fine-tuned with **LoRA** for Arabic legal NLP, significantly improving **lexical and semantic alignment**.

## **Setup**

### **Installation**

```sh
pip install transformers torch datasets accelerate
```

### **Inference**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mbayan/Llama-3.2-3b-ArLJP"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

instruction = "استنادًا إلى الوقائع، قم بتحليل الأسباب واستخرج نص الحكم النهائي"
legal_case = """
الوقائع:
تقدم خالد الأحمدي بدعوى ضد ماجد الزهراني للمطالبة باسترداد مبلغ (٥٠,٠٠٠) ريال، الذي دفعه له للاستثمار في تجارة إلكترونية، إلا أن المشروع لم يُنفذ ولم يُرد المبلغ.

الأسباب:
بما أن النزاع يتعلق باسترداد مبلغ مالي، فهو من اختصاص المحكمة التجارية. لكن نظرًا لوفاة المدعي قبل الجلسة الأولى، فإن الدعوى تنقطع ويجب تقديمها من الورثة وفقًا للنظام.

"""
input_text = f"{instruction}\n\n{legal_case}"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## **Fine-Tuning & Evaluation**

Fine-tuned using **LoRA** with **75 diverse instructions** for enhanced **generalization and adaptability**.  
 **Evaluation Metrics:** BLEU, ROUGE, BERTScore, and qualitative legal assessments.

| Model                       | ROUGE-1 | ROUGE-2 | BLEU | BERTScore |
| --------------------------- | ------- | ------- | ---- | --------- |
| **LLaMA-3.2-3B-Instruct**   | 0.08    | 0.02    | 0.01 | 0.54      |
| **LLaMA-3.2-3B-Fine-Tuned** | 0.50    | 0.39    | 0.24 | 0.74      |
| **LLaMA-3.1-8B-Fine-Tuned** | 0.53    | 0.41    | 0.26 | 0.76      |

## **Citation**

If you use this dataset or models, please cite:

```bibtex
@misc{kmainasi2025largelanguagemodelspredict,
      title={Can Large Language Models Predict the Outcome of Judicial Decisions?},
      author={Mohamed Bayan Kmainasi and Ali Ezzat Shahroor and Amani Al-Ghraibah},
      year={2025},
      eprint={2501.09768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.09768},
}
```

## **License**

This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
