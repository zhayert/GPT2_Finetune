from transformers import GPT2Tokenizer,GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained("../GPT2_pretrained")
model = GPT2Model("../GPT2_pretrained")
print(tokenizer)
print(model)