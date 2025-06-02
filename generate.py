from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("finetuned_gemma", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("finetuned_gemma")

def chat(query):
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

chat("Explain the difference between BFS and DFS.")
