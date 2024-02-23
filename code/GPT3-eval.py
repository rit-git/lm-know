import openai
import torch
import warnings
import random
import math
import os
import scipy.stats
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_gpt3():
    with open(os.path.join(ROOT_DIR, 'openai-key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key

def random_state(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def read_json(path):
	test_data = []
	for filename in os.listdir(path):
		with open(os.path.join(path, filename), 'r') as f:
			for line in f:
				dict_temp = json.loads(line)
				if "masked_sentence" in dict_temp:
					doc = dict_temp["masked_sentence"]
					obj = dict_temp["obj_surface"]
					sub = dict_temp["sub_surface"]
					test_data.append([sub, obj, doc])

				elif "evidences" in dict_temp:
					for sam in dict_temp["evidences"]:
						doc = sam["masked_sentence"]
						obj = sam["obj_surface"]
						sub = sam["sub_surface"]
						test_data.append([sub, obj, doc])
				
	print(len(test_data))
	return test_data


def generate_prompt(ph):
	prompt = ph
	return prompt


def entropy_cal(dict_o, dict_i):
	if len(dict_o) == 0:
		temp = []
		p_all = 0
		for tok in dict_i:
			prob = math.exp(dict_i[tok])
			p_all += prob
			temp += [prob]
		temp += [1-p_all]
		return scipy.stats.entropy(temp)
	elif len(dict_i) == 0:
		temp = []
		p_all = 0
		for tok in dict_o:
			prob = math.exp(dict_o[tok])
			p_all += prob
			temp += [prob]
		temp += [1-p_all]
		return scipy.stats.entropy(temp)
	else:
		temp_o, temp_i = [], []
		p_all_o, p_all_i = 0, 0
		for tok in dict_i:
			if tok in dict_o:
				prob_i = math.exp(dict_i[tok])
				prob_o = math.exp(dict_o[tok])
				p_all_o += prob_o
				p_all_i += prob_i
				temp_o += [prob_o]
				temp_i += [prob_i]
		temp_i += [1-p_all_i]
		temp_o += [1-p_all_o]
		return scipy.stats.entropy(temp_o) - scipy.stats.entropy(temp_i)
	return score


def calcul_score(dict_orig, dict_inject):
	avg_score = 0.0 
	num = 0
	for i in range(max(len(dict_inject), len(dict_orig))):
		if i >= len(dict_inject):
			avg_score += entropy_cal(dict_orig[i], {})
		elif i >= len(dict_orig):
			avg_score += entropy_cal({}, dict_inject[i])
		else:
			avg_score += entropy_cal(dict_orig[i], dict_inject[i])

		num +=1
	return avg_score/num


def gpt3_paraphrase(samples):
	""" Using GPT-3 to paraphrase"""

	setup_gpt3()
	dict_para = {}
	for pair in samples:
		obj = pair[1]
		ph = pair[0]
		inject = ph + ' ' + obj + '. '
		prompt = generate_prompt(ph)
		output = openai.Completion.create(
		  engine="davinci",
		  prompt=prompt,
		  max_tokens=5,
		  n = 1,
		  logprobs = 100,#100
		  temperature= 0.0, 
		  top_p= 1,
		  stop='\n'
		)

		dict_orig = output["choices"][0]["logprobs"]["top_logprobs"]

		prompt = generate_prompt(inject + ph)
		output = openai.Completion.create(
		  engine="davinci",
		  prompt=prompt,
		  max_tokens=5,
		  n = 1,
		  logprobs = 100,#100
		  temperature= 0.0, 
		  top_p= 1,
		  stop='\n'
		)
		dict_inject = output["choices"][0]["logprobs"]["top_logprobs"]

		score = calcul_score(dict_orig, dict_inject)

		print(score)

	return dict_para

samps = [("Barack Obama is married to", "Michelle Obama")]
gpt3_paraphrase(samps)