from sys import argv

# additionally required packages
# peft,transformers,modelscope
from colorama import Fore,Style
import torch
try:
    import torch_directml  # type: ignore

    device = torch_directml.device() # GPU 0 (intel UHD Graphics 630)
    print(Fore.GREEN + 'Using DirectML for GPU acceleration')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device,f'{type(device)}')
torch_directml._initialized = True

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
from modelscope import snapshot_download, AutoTokenizer


import os

modelDict={'basicModel':{'model':'./Nya3/', 'tokenizer':'./Nya3/tokenizer/'}}
def loadModel(modeldict:dict):
    global model, tokenizer
    model=AutoModelForCausalLM.from_pretrained(modeldict['model'], torch_dtype=torch.float16)
    tokenizer=AutoTokenizer.from_pretrained(modeldict['tokenizer'])
    model.eval()
    model.to(device)
def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
 
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
 
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
def ask(question): 
    messages = [
            {"role": "user", "content":question }
        ]
    print(Fore.YELLOW,question,Fore.RESET,sep='\n')
    answer = predict(messages, model, tokenizer)
    print(Fore.GREEN,answer,Fore.RESET,sep='\n')
    return answer

import flask
app = flask.Flask(__name__)

@app.route('/<modelName>/<question>', methods=['POST'])
def get_answer(modelName, question):
    print(modelName, question)
    if modelName in modelDict:
        loadModel(modelDict[modelName])
    answer = ask(question)
    return flask.jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
