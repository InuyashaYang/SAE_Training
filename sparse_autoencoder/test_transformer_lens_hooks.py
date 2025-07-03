from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
for name, module in model.named_modules():
    print(name)
