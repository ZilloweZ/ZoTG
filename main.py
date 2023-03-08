from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = FastAPI()

class Message(BaseModel):
    message: str

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

@app.post('/chatbot')
def chatbot(message: Message):
    input_ids = tokenizer.encode(message.message, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {'response': response}
