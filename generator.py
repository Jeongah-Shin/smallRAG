from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)


input_text = "summarize: This is a test document about Tensorflow and RAG models."
input_ids = tokenizer(input_text, return_tensors="tf").input_ids

outputs = model.generate(input_ids, max_length=50, num_beams=3, early_stopping=True)
print("Generated Text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
