import os
import openai

from openai import OpenAI
import backoff 
from transformers import GPT2Tokenizer
import warnings
completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

def tokens_in_text(text):
    """
    Accurately count the number of tokens in a string using the GPT-2 tokenizer.
    
    :param text: The input text.
    :return: The exact number of tokens in the text.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
    return len(tokens)

#api_key = os.getenv("OPENAI_API_KEY", "")
api_key = "sk-0fdda24603944d1cb45bafcc21aee206"
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
#api_base = os.getenv("OPENAI_API_BASE", "")
api_base = "https://api.deepseek.com"
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base




@backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(**kwargs):
    #client = OpenAI(api_key="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", base_url="https://api.deepseek.com")
    client = OpenAI(api_key="sk-0fdda24603944d1cb45bafcc21aee206", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(**kwargs)
    #return client.chat.completions.create(**kwargs)
    #print("Here is the response\n")
    #print(response)
    return response

def gpt(prompt, model="deepseek-chat", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    

def chatgpt(messages, model="deepseek-chat", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1, stop=stop)
        #outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    elif backend == "deepseek-chat":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
