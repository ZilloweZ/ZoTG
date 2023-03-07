from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPT2TokenizerFast

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained(
    'gpt2', pad_token_id=tokenizer_gpt2.eos_token_id)

tokenizer_gptneo = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B")
model_gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
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
        return JsonResponse({'response': response})
    else:
        return render(request, 'chatbot.html')
