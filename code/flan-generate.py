from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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
		if num_process %50 == 0:
			print(num_process, num_bad)

		prompt = "Write a paragraph about {}.\n Paragraph:".format(sub) 

		prompt = tokenizer(prompt, return_tensors="pt").to(device)

		gen_tokens = model.generate(**prompt, max_length=500, early_stopping = False, eos_token_id =None)
		gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]



		if len(gen_text)<20:
			num_bad += 1

		dict_out[sub] = {'gen_paragraph': gen_text, 'Human':test[sub]['Human']}

		num_process += 1

	print(num_process, num_bad)


	return dict_out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
path = "data/entities.json"
data = read_json(path)
print(len(data))
dict_out = main(data)

with open('out/generated_paragraphs_flan.json', 'w') as f:
    json.dump(dict_out, f)