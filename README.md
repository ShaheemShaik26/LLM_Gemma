# LLM_Gemma
a small LLM project of fine-tuning Gemma-2b of Google

# Gemma(DSA & DBMS) - LLM
This project is my first attempt at fine-tuning a large language model (LLM) called Gemma-2B to help explain Data Structures & Algorithms (DSA) and Database Management Systems (DBMS) topics. 
I created a small dataset with questions and answers about DSA and DBMS and used LoRA to fine-tune the Gemma-2B model on this data. The idea is that this model can act like a simple tutor that understands and answers basic questions about these subjects. I have used a very small dataset of 20 values, so the model is very biased with answers.
I am still learning how LLMs work and how to train them properly, so this is a beginner-friendly experiment. The training code and generation code are included.

# How to run
1. Install dependencies:
>> pip install -r requirements.txt

2. Run the training script:
>> python train.py

3. After training, run the generation script to chat with the model:
>> python generate.py

## Explanation of code
- "train.py" loads the Gemma-2B model and tokenizer, prepares the dataset, and fine-tunes the model using LoRA.
- "generate.py" loads the fine-tuned model and lets you input questions interactively, then prints the model's answer.
- "data/dsa_dbms_dataset.jsonl" contains example question-answer pairs used for training.
