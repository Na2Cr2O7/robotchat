MODELDIR='qwen/Qwen2.5-0.5B-Instruct'
LORADIR='/loraMerge'
SAVEDIR='./nya3'
SAVEDIR_TOKENIZER=SAVEDIR +'/tokenizer'
import os
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)
if not os.path.exists(SAVEDIR_TOKENIZER):
    os.makedirs(SAVEDIR_TOKENIZER)

from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
modelpath=snapshot_download(MODELDIR,cache_dir='./BASEMODEL')
tokenizer=AutoTokenizer.from_pretrained(modelpath)
model=AutoModelForCausalLM.from_pretrained(modelpath)
loraModel=PeftModel(model,LORADIR,torch_dtype=torch.float16)
model=loraModel.merge_and_unload()
model.save_pretrained(SAVEDIR)
tokenizer.save_pretrained(SAVEDIR_TOKENIZER)