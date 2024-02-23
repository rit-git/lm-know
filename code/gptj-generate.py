from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
import random
import math
import os
import scipy.stats
import json
from scipy.special import rel_entr
warnings.filterwarnings("ignore")


def read_json(path):
	f = open(path)
	test_data = json.load(f)
	return test_data


def main(test):
	dict_out = {}
	num_process = 0
	num_bad = 0, 0 

	for sub in test:
		if num_process %10 == 0:
			print(num_process, num_bad)

		print(sub)

		prompt = "Write a paragraph with only five sentences about {}.\n Paragraph:".format(sub) 

		prompt_enc = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

		gen_tokens = model.generate(prompt_enc,do_sample=True,temperature=0.9,max_length=100)
		gen_text = tokenizer.batch_decode(gen_tokens)[0]


		print(gen_text)

		# if len(ans_orig)<20:
		# 	num_bad += 1

		dict_out[sub] = {'gen_paragraph': ans_orig, 'Human':test[sub]['Human']}

		num_process += 1

	print(num_process, num_bad)


	return dict_out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
path = "data/entities.json"
data = read_json(path)
print(len(data))
dict_out = main(data)

with open('out/generated_paragraphs.json', 'w') as f:
    json.dump(dict_out, f)