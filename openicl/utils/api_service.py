import json
import requests
import torch
import numpy as np
from tqdm import tqdm

API_NAME_LIST = ['opt-175b', 'chatgpt']
API_REQUEST_CONFIG = {
    'opt-175b': {
        'URL' : "http://10.140.1.159:6010/completions",
        'headers' : {
            "Content-Type": "application/json; charset=UTF-8"
        }
    }
}
PROXIES = {"https": "", "http": ""}


def is_api_available(api_name):
    if api_name == None:
        return False
    return True if api_name in API_NAME_LIST else False


def api_get_ppl(api_name, input_texts):
    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 100, "echo": True}
        response = json.loads(
                requests.post(API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload), headers=API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
        loss_lens = np.array([len(r['logprobs']['token_logprobs']) for r in response['choices']])
        
        loss = [r['logprobs']['token_logprobs'] for r in response['choices']]

        max_len = loss_lens.max()
        loss_pad = list(map(lambda l: l + [0] * (max_len - len(l)), loss))
        loss = -np.array(loss_pad)

        loss = torch.tensor(loss)
        ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
        return ce_loss / lens


def api_get_tokens(
    api_name, input_texts, src_lang, tgt_lang,
    api_key_path=None, rpm=20
):
    length_list = [len(text) for text in input_texts]
    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 100, "echo": True}
        response = json.loads(
                requests.post(API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload), headers=API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        return [r['text'][length:] for r, length in zip(response['choices'], length_list)]
    elif api_name == "chatgpt":
        import openai
        import time
        from threading import Thread, Lock
        openai.api_key_path = api_key_path
        generated = {}

        def requestAPI(src_lang, tgt_lang, idx, context, lock):
            prompts = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that translates {} to {}.".format(src_lang, tgt_lang)
                },
                {
                    "role": "user",
                    "content": context
                }
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompts
            )['choices'][0]['message']['content']
            lock.acquire()
            generated[idx] = response
            lock.release()

        def trying(src_lang, tgt_lang, input_texts, test_ids):
            threads = []
            lock = Lock()
            for idx in tqdm(test_ids, desc="Batch"):
                text = input_texts[idx]
                thread = Thread(target=requestAPI, args=(src_lang, tgt_lang, idx, text, lock))
                thread.start()
                threads.append(thread)
                time.sleep(80 / rpm)
            for thread in threads:
                if thread.is_alive():
                    thread.join()
            retry = []
            for idx in test_ids:
                if not idx in generated:
                    retry.append(idx)
            return retry
        
        retry = range(len(input_texts))
        while True:
            retry = trying(src_lang, tgt_lang, input_texts, retry)
            if len(retry) == 0:
                break
            else:
                print(f"retrying dead ids: {retry}")
        generated = dict(sorted(generated.items(), key=lambda x: int(x[0]))).values()
        return generated