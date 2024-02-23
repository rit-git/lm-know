import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
# from parrot import Parrot
# from parapherase import para
import logging
import json
# from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import random
import scipy.stats
from scipy.special import rel_entr
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)# OPTIONAL
import os
import warnings
warnings.filterwarnings("ignore")

random.seed(1234)

def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def read_json(path):
    documents = []
    with open(path) as f:
        for line in f:
            dict_line = json.loads(line) 
            documents += [dict_line]
    return documents

# model.to('cuda')  # if you have gpu

def corr_score(ll):
	# sort_l = np.argsort(ll)
	score = 0.0
	nn = 0
	for i in range(len(ll)-1):
		# if i != sort_l[i]:
		# if ll[i]>=ll[i+1]:
		# 	score += 1
		for j in range(i+1, len(ll)):
			if ll[i]<ll[j]:
				score += 1
			nn += 1
	return score/nn


def predict_unknown(test_data, top_k=100):
    # Tokenize input
	dict_rel = {}
	predict_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
	num = 0

	for samp in test_data:
		sub = samp['sub']
		obj = samp['obj']
		rel = samp['rel']
		text = samp['prompt']

		mask_str = " ".join(['[MASK]']*len(tokenizer.encode(obj,add_special_tokens=False,return_tensors='pt')[0]))

		prompt = text.replace('[Y]', mask_str)

		label = text.replace('[Y]', obj)
		text_t = "[CLS] %s [SEP]"%prompt
		label = "[CLS] %s [SEP]"%label
		tokenized_text = tokenizer.tokenize(text_t)
		tokenized_label = tokenizer.tokenize(label)

		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
		tokens_tensor = torch.tensor([indexed_tokens])
		label_tensor = torch.tensor([indexed_label])
		predict_mask = tokens_tensor.eq(predict_token_id)

		label_ind = label_tensor[predict_mask]
		with torch.no_grad():
			outputs = model(tokens_tensor)
			predictions = outputs[0]

		probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
		probs = probs.numpy()

		rank = 0.0
		for k in range(len(probs)):

			val_l = probs[k][label_ind[k]]

			prob = sorted(probs[k], reverse = True)
			rank += prob.index(val_l)+1

		rank = rank/len(probs)

		if rank>100:
			if rel not in dict_rel:
				dict_rel[rel] = [(sub, obj)]
			else:
				dict_rel[rel] += [(sub, obj)]

		num += 1
		if num % 200 == 0:
			print(num)

	print(len(dict_rel))
	for rel in dict_rel:
		print(len(dict_rel[rel]))

	with open('data/dict_pair.json', 'w') as outfile:
		json.dump(dict_rel, outfile)




def predict_all(test_data, top_k=100):
    # Tokenize input
	acc = 0.0
	num = 0

	ent1_list = []
	ent2_list = []
	kl1_list = []
	kl2_list = []
	rank_list = []
	predict_token_id = tokenizer.convert_tokens_to_ids('[MASK]')

	for samp in test_data:
		sub = samp['sub']
		obj = samp['obj']
		text = samp['prompt']
		# print(sub, obj, text)

		mask_str = " ".join(['[MASK]']*len(tokenizer.encode(obj,add_special_tokens=False,return_tensors='pt')[0]))

		prompt = text.replace('[Y]', mask_str)


		label = text.replace('[Y]', obj)
		label = "[CLS] %s [SEP]"%label
		tokenized_label = tokenizer.tokenize(label)
		indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
		label_tensor = torch.tensor([indexed_label])


		b = text.replace('[Y]', obj)

		##### subject prediction

		# mask_str = " ".join(['[MASK]']*len(tokenizer.encode(sub,add_special_tokens=False,return_tensors='pt')[0]))

		# prompt = text.replace(sub, mask_str)
		# text = text.replace('[Y]', obj)
		# prompt = prompt.replace('[Y]', obj)

		# label = text
		# label = "[CLS] %s [SEP]"%label
		# tokenized_label = tokenizer.tokenize(label)
		# indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
		# label_tensor = torch.tensor([indexed_label])


		# b = text

		####################



		text_t = "[CLS] " + b + " [SEP] " + prompt + " [SEP]"
		tokenized_text = tokenizer.tokenize(text_t)
		if len(tokenized_text)<=512:# and len(l_t)==1:

			masked_index = tokenized_text.index("[MASK]")
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
			tokens_tensor = torch.tensor([indexed_tokens])
			predict_mask = tokens_tensor.eq(predict_token_id)
			# tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu
			# gold_list += [label]
			with torch.no_grad():
				outputs = model(tokens_tensor)
				predictions = outputs[0]

			probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
			prob_inject = probs.numpy()
			ent2 = 0.0
			for mask_pred in prob_inject:
				ent2 += scipy.stats.entropy(mask_pred)
			ent2 = ent2/len(prob_inject) 


			text_t = "[CLS] %s [SEP]"%prompt
			tokenized_text = tokenizer.tokenize(text_t)

			masked_index = tokenized_text.index("[MASK]")
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
			tokens_tensor = torch.tensor([indexed_tokens])
			predict_mask = tokens_tensor.eq(predict_token_id)
			num+=1
			# tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu
			# gold_list += [label]
			with torch.no_grad():
				outputs = model(tokens_tensor)
				predictions = outputs[0]

			probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
			prob_orig = probs.numpy()

			ent1 = 0.0
			for mask_pred in prob_orig:
				ent1 += scipy.stats.entropy(mask_pred)
			ent1 = ent1/len(prob_orig)

			ent1_list += [ent1-ent2]

			with torch.no_grad():
				outputs = model_fine(tokens_tensor)
				predictions = outputs[0]

			probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
			prob_fine = probs.numpy()
			ent3 = 0.0
			for mask_pred in prob_fine:
				ent3 += scipy.stats.entropy(mask_pred)

			ent3 = ent3/len(prob_fine)
			ent2_list += [ent1-ent3]

			# if num >600:

			# 	print(sub, obj, text)
			# 	print(label_tensor, predict_mask)
			label_ind = label_tensor[predict_mask]
			rank = 0.0
			for k in range(len(prob_orig)):

				val_l = prob_orig[k][label_ind[k]]

				prob_s = sorted(prob_orig[k], reverse = True)
				rank += prob_s.index(val_l)+1

			rank = rank/len(prob_orig)
			rank_list += [rank]


			KL = 0.0
			for k in range(len(prob_orig)):
				KL += sum(rel_entr(prob_orig[k], prob_inject[k]))
			KL = KL/len(prob_orig)
			kl1_list += [KL]


			KL = 0.0
			for k in range(len(prob_orig)):
				KL += sum(rel_entr(prob_orig[k], prob_fine[k]))
			KL = KL/len(prob_orig)
			kl2_list += [KL]


			# if prob.index(val_l) > 0 and abs(ent1-ent2)<0.1:
			# 	list_bad += [{'doc':text, 'obj':label, 'sub':sub, 'rel':rel}]


		if num % 100 == 0:
			print(num)

	np.save('out/vanila_ent1_sub.npy', ent1_list)
	np.save('out/vanila_ent2_sub.npy', ent2_list)
	np.save('out/vanila_rank_sub.npy', rank_list)
	np.save('out/vanila_kl1_sub.npy', kl1_list)
	np.save('out/vanila_kl2_sub.npy', kl2_list)




def predict_masked_sent(path_dict, top_k=100):
    # Tokenize input
	acc = 0.0
	num = 0
	rr, e1, e2, kl = 0.0, 0.0, 0.0, 0.0

	ent1_list = []
	ent2_list = []
	rank_list = []
	KL_list = []
	list_bad = []

	f = open(path_dict)
	dict_test = json.load(f)
	predict_token_id = tokenizer.convert_tokens_to_ids('[MASK]')

	for rel in dict_test:
		list_samp = dict_test[rel]
		orders = range(len(list_samp))
		temp_ent1 = []
		temp_ent2 = []
		temp_rank = []
		temp_KL = []
		for samp in list_samp:
			sub = samp['sub']
			obj = samp['obj']
			text = samp['prompt']

			mask_str = " ".join(['[MASK]']*len(tokenizer.encode(obj,add_special_tokens=False,return_tensors='pt')[0]))

			prompt = text.replace('[Y]', mask_str)


			label = text.replace('[Y]', obj)
			label = "[CLS] %s [SEP]"%label
			tokenized_label = tokenizer.tokenize(label)
			indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
			label_tensor = torch.tensor([indexed_label])


			b = text.replace('Y', obj)

			text_t = "[CLS] " + b + " [SEP] " + prompt + " [SEP]"
			tokenized_text = tokenizer.tokenize(text_t)
			if len(tokenized_text)<=512:# and len(l_t)==1:

				masked_index = tokenized_text.index("[MASK]")
				indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				tokens_tensor = torch.tensor([indexed_tokens])
				predict_mask = tokens_tensor.eq(predict_token_id)
				# tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu
				# gold_list += [label]
				with torch.no_grad():
					outputs = model(tokens_tensor)
					predictions = outputs[0]

				probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
				prob_inject = probs.numpy()
				ent2 = 0.0
				for mask_pred in prob_inject:
					ent2 += scipy.stats.entropy(mask_pred)
				ent2 = ent2/len(prob_inject) 


				text_t = "[CLS] %s [SEP]"%prompt
				tokenized_text = tokenizer.tokenize(text_t)

				masked_index = tokenized_text.index("[MASK]")
				indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				tokens_tensor = torch.tensor([indexed_tokens])
				predict_mask = tokens_tensor.eq(predict_token_id)
				num+=1
				# tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu
				# gold_list += [label]
				with torch.no_grad():
					outputs = model(tokens_tensor)
					predictions = outputs[0]

				probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
				prob_orig = probs.numpy()
				ent1 = 0.0
				for mask_pred in prob_orig:
					ent1 += scipy.stats.entropy(mask_pred)
				ent1 = ent1/len(prob_orig)

				ent1_list += [abs(ent1-ent2)]
				temp_ent1 += [abs(ent1-ent2)]

				# with torch.no_grad():
				# 	outputs = model_fine(tokens_tensor)
				# 	predictions = outputs[0]

				# probs = torch.nn.functional.softmax(predictions.unsqueeze(0)[0, predict_mask], dim=-1)
				# prob = probs.numpy()
				# ent3 = 0.0
				# for mask_pred in prob:
				# 	ent3 += scipy.stats.entropy(mask_pred)
				# ent3 = ent3/len(prob)
				# ent2_list += [ent1-ent3]
				# temp_ent2 += [ent1-ent3]
				label_ind = label_tensor[predict_mask]
				rank = 0.0
				for k in range(len(prob_orig)):

					val_l = prob_orig[k][label_ind[k]]

					prob_s = sorted(prob_orig[k], reverse = True)
					rank += prob_s.index(val_l)+1

				rank = rank/len(prob_orig)
				rank_list += [rank]
				temp_rank += [rank]

				KL = 0.0
				for k in range(len(prob_orig)):
					KL += sum(rel_entr(prob_orig[k], prob_inject[k]))
				KL = KL/len(prob_orig)
				temp_KL += [KL]
				KL_list += [KL]


				# if prob.index(val_l) > 0 and abs(ent1-ent2)<0.1:
				# 	list_bad += [{'doc':text, 'obj':label, 'sub':sub, 'rel':rel}]


			if num % 50 == 0:
				print(num)

		# print(list_samp)
		# print(temp_rank, corr_score(temp_rank))
		# print(temp_ent1, corr_score(temp_ent1))
		# print(temp_KL, corr_score(temp_KL))
		# print('------------------------')


		if len(temp_ent1)>0:
			rr += corr_score(temp_rank)
			e1 += corr_score(temp_ent1)
			kl += corr_score(temp_KL)
			# e2 += corr_score(temp_ent2)
	np.save('out/vanila_ent1.npy', ent1_list)
	# np.save('out/vanila_ent2.npy', ent2_list)
	np.save('out/vanila_rank.npy', rank_list)
	np.save('out/vanila_KL.npy', KL_list)
	print('r:', rr/len(dict_test), 'e1:', e1/len(dict_test), 'KL:', kl/len(dict_test), "num:", num)

def predict_masked_para(test, top_k=100):
    # Tokenize input
	acc = 0.0
	num = 0
	ent_list = []
	rank_list = []

	for text, label in test:
		if "[MASK]" in text:

			ind_p = tokenizer.convert_tokens_to_ids(label)
			l_t = tokenizer.tokenize(label)


			b = text.replace('[MASK]', label)
			para_phrases = parrot.augment(input_phrase=b)

			text_t = "[CLS] %s [SEP]"%text
			tokenized_text = tokenizer.tokenize(text_t)

			if len(tokenized_text)<=512:

				masked_index = tokenized_text.index("[MASK]")
				indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				tokens_tensor = torch.tensor([indexed_tokens])
				num+=1
				# tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu
				# gold_list += [label]
				with torch.no_grad():
					outputs = model(tokens_tensor)
					predictions = outputs[0]

				probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
				prob = probs.numpy()
				ent1 = scipy.stats.entropy(prob) 

				if para_phrases is not None:
					if len(para_phrases)>=3:
						temp = []

						for bb in [para_phrases[-1][0], para_phrases[int(len(para_phrases)/2)][0], para_phrases[0][0]]:

						
							text_t = "[CLS] " + bb + " [SEP] " + text + " [SEP]"
							tokenized_text = tokenizer.tokenize(text_t)
							if len(tokenized_text)>512 or len(l_t)!=1:
								continue


							masked_index = tokenized_text.index("[MASK]")
							indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
							tokens_tensor = torch.tensor([indexed_tokens])
							# tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu
							# gold_list += [label]
							with torch.no_grad():
								outputs = model(tokens_tensor)
								predictions = outputs[0]

							probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
							prob = probs.numpy()
							ent2 = scipy.stats.entropy(prob) 

							temp += [ent1-ent2]


						if len(temp) == 3:
							ent_list += [temp]

							val_l = prob[ind_p]

							prob = sorted(prob, reverse = True)
							rank_list += [prob.index(val_l)+1]

			if num % 200 == 0:
				print(num)
				print(len(ent_list))
				for k in range(3):
					ee = [samp[k] for samp in ent_list]
					ee = np.array(ee)
					m, b = np.polyfit(ee, rank_list, 1)
					plt.scatter(ee, rank_list, s= 5, alpha = 0.8)
					plt.plot(ee, m*ee+b, color='red')
					# plt.yscale('log')
					plt.xlabel("Entropy Change")
					plt.ylabel("Rank")
					plt.show()

			# ind_p = tokenizer.convert_tokens_to_ids('great')
			# ind_n = tokenizer.convert_tokens_to_ids('terrible')
			# p_s = probs[ind_p].item()
			# n_s = probs[ind_n].item() 
			# if p_s>=n_s:
			# 	pred_list += [1]
			# else:
			# 	pred_list += [0]				

	# print('accuracy:', accuracy_score(gold_list, pred_list))
	# print('F1:', f1_score(gold_list, pred_list))

random_state(1234)
# parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

model_fine = BertForMaskedLM.from_pretrained('bert-base-uncased')
model_fine.load_state_dict(torch.load("checkpoint/model.pt"))	
model_fine.eval()
path_dict = "data/dict_test.json"       
# test = read_json('TREx/')
predict_masked_sent(path_dict)
test_data = read_json('data/test.jsonl')
predict_all(test_data)
