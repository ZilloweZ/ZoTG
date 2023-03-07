from flask import Flask, request

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPT2TokenizerFast

app = Flask(__name__)

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

model_gpt2 = GPT2LMHeadModel.from_pretrained(

        'gpt2', pad_token_id=tokenizer_gpt2.eos_token_id)

tokenizer_gptneo = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B")

model_gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

@app.route('/chatbot', methods=['POST'])

def chatbot():

        message = request.form['message']

        input_ids_gpt2 = tokenizer_gpt2.encode(message, return_tensors='pt')

        output_gpt2 = model_gpt2.generate(input_ids_gpt2,

                                          max_length=1000,

                                          do_sample=True)

        response_gpt2 = tokenizer_gpt2.decode(output_gpt2[0],

                                              skip_special_tokens=True)

        input_ids_gptneo = tokenizer_gptneo.encode(message,

                                                   return_tensors='pt')

        output_gptneo = model_gptneo.generate(input_ids_gptneo,

                                              max_length=1000,

                                              do_sample=True)

        response_gptneo = tokenizer_gptneo.decode(output_gptneo[0],

                                                  skip_special_tokens=True)

        response = f"GPT-2: {response_gpt2}\n\nGPT-Neo: {response_gptneo}"

        return response

if __name__ == '_main_':

        app.run(debug=True)

