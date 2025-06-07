from sys import argv

# additionally required packages
# peft,transformers,modelscope
from colorama import Fore,Style
import torch
try:
    import torch_directml  # type: ignore

    device = torch_directml.device(0) # GPU 0 (intel HD Graphics 520)
    print(Fore.GREEN + 'Using DirectML for GPU acceleration')
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch_directml._initialized = True


device='cpu'
print('Using device:', device,f'{type(device)}')



from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
from modelscope import snapshot_download, AutoTokenizer
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
import json
import os

def splitThink(text:str)->list:
    r=text.find('</think>')
    left=text[:r]
    right=text[r+8:]
    return [left,right]
def containsThink(text:str)->bool:
    return text.find('</think>')!=-1
modelDict={'basicModel_UNOBLITERATED':{'model':'./nya15/', 'tokenizer':'./nya15/tokenizer/','lora':None},'basicModel':{'model':'./nya17/', 'tokenizer':'./nya17/','lora':None}}
oldModelName=''
def loadModel(modelName,modeldict:dict):
    global model, tokenizer
    if modelName==oldModelName:
        return
    model=AutoModelForCausalLM.from_pretrained(modeldict['model'], torch_dtype=torch.float16)
    if modeldict['lora'] is not None:
        model=PeftModel.from_pretrained(model,modeldict['lora'],torch_dtype=torch.float16)

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
    global model
    if model is None:
        loadModel('basicModel',modelDict['basicModel'])
    messages = [
            {"role": "user", "content":question }
        ]
    print(Fore.YELLOW,question,Fore.RESET,sep='\n')
    answer = predict(messages, model, tokenizer)
    if containsThink(answer):
        left,right=splitThink(answer)
        print(Fore.YELLOW,left,Fore.RESET,sep='\n')
        answer=ask(right)
    print(Fore.GREEN,answer,Fore.RESET,sep='\n')
    return answer

import flask
app = flask.Flask(__name__)

@app.route('/<modelName>/<question>', methods=['POST'])
def get_answer(modelName, question):
    print(modelName, question)
    if modelName in modelDict:
        loadModel(modelName,modelDict[modelName])
    answer = ask(question)
    dt=json.dumps({'question':question,'answer':answer})
    with open('replyHistory.txt','a') as f:
        f.write(dt)
    return flask.jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
