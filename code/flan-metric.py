from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import warnings
import random
import math
import os
import scipy.stats
import json
import numpy as np
from scipy.special import rel_entr
warnings.filterwarnings("ignore")


def read_json(path):
	test_data = []
	with open(path, 'r') as f:
		for line in f:
			dict_temp = json.loads(line)
			test_data += [dict_temp]

	return test_data


def calcul_score(pred_orig, pred_inject):
	avg_score_e, avg_score_kl  = 0.0, 0.0
	num = 0
	num_kl = 0
	zero = np.zeros((pred_orig[0].shape[1]))
	for i in range(max(len(pred_inject), len(pred_orig))):
		if i >= len(pred_inject):
			avg_score_e += abs(scipy.stats.entropy(pred_orig[i][0]))
			# avg_score_kl += sum(rel_entr(pred_orig[i].detach().cpu().numpy(), zero))
		elif i >= len(pred_orig):
			avg_score_e += abs(scipy.stats.entropy(pred_inject[i][0]))
			# avg_score_kl += sum(rel_entr(zero, pred_inject[i].detach().cpu().numpy()))
		else:
			avg_score_e += abs(scipy.stats.entropy(pred_orig[i][0]) - scipy.stats.entropy(pred_inject[i][0]))
			avg_score_kl += sum(rel_entr(pred_orig[i][0], pred_inject[i][0]))
			num_kl += 1

		num +=1

	if num == 0:
		ent = 0
	else:
		ent = avg_score_e/num

	if num_kl == 0:
		kl = 0
	else:
		kl = avg_score_kl/num_kl



	return ent, kl


def flan_know(samp):
	obj = samp['obj']
	obj_list = obj.split(' ')
	fact = samp['prompt']
	prompt = 'Question: ' + samp['prompt-q'] + ' Answer:'
	inject_prompt = 'Question: knowing the fact that ' + fact[:-1] + ', ' + samp['prompt-q'] + ' Answer:'


	prompt = tokenizer(prompt, return_tensors="pt").to(device)

	gen_tokens = model.generate(**prompt, max_length=5, output_scores=True, return_dict_in_generate=True)
	gen_orig = tokenizer.batch_decode(gen_tokens.sequences, skip_special_tokens=True)
	pred_orig = []
	for logit in gen_tokens.scores:
		if torch.argmax(logit)>3:
			pred_orig += [soft(logit).detach().cpu().numpy()]


	inject_prompt = tokenizer(inject_prompt, return_tensors="pt").to(device)

	gen_tokens = model.generate(**inject_prompt, max_length=5, output_scores=True, return_dict_in_generate=True)
	gen_inject = tokenizer.batch_decode(gen_tokens.sequences, skip_special_tokens=True)
	pred_inject = []
	for logit in gen_tokens.scores:
		if torch.argmax(logit)>3:
			pred_inject += [soft(logit).detach().cpu().numpy()]

	score_e, score_kl = calcul_score(pred_orig, pred_inject)

	return score_e, score_kl, pred_orig, pred_inject, gen_orig, gen_inject


def main(data, out):
	num_process = 0
	for samp in data:
		if num_process< len(out):
			num_process += 1
			continue 
		temp = {}
		if num_process %200 == 0:
			with open('out/final_score_flan.jsonl', 'w') as outfile:
			    for entry in out:
			        json.dump(entry, outfile)
			        outfile.write('\n')
			print(num_process)

		score_e, score_kl, dict_orig, dict_inject, gen_orig, gen_inject = flan_know(samp)
		dict_orig = [i.tolist() for i in dict_orig]
		dict_inject = [i.tolist() for i in dict_inject]
		temp['kl'], temp['entropy'] = score_kl, score_e
		# temp['pred_orig'], temp['pred_inject'] = dict_orig, dict_inject
		temp['gen_orig'], temp['gen_inject'] = gen_orig, gen_inject
		for kk in samp:
			temp[kk] = samp[kk]

		out += [temp]
		num_process += 1

	return out


torch.cuda.empty_cache()
path = "data/filtered_pred_flan.jsonl"
data = read_json(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)
soft = torch.nn.Softmax(dim=1)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
print(len(data))
if os.path.isfile('out/final_score_flan.jsonl'):
	out = read_json('out/final_score_flan.jsonl')
else:
	out = []

print(len(out))
out = main(data, out)


with open('out/final_score_flan.jsonl', 'w') as outfile:
    for entry in out:
        json.dump(entry, outfile)
        outfile.write('\n')


