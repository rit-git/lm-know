import numpy as np
import random
import scipy.stats
import matplotlib.pyplot as plt
import json
random.seed(1234)

def read_jj(path):
    dict_test = {}
    with open(path) as f:
        for line in f:
            dict_line = json.loads(line) 
            prompt = dict_line["template"]
            type_p = dict_line['type']
            dict_test[prompt] = type_p

    return dict_test

def read_json(path):
    documents = []
    dict_test = {}
    with open(path) as f:
        for line in f:
            dict_line = json.loads(line) 
            prompt = dict_line['prompt'].replace(dict_line['sub'], '[X]')
            dict_line['prompt'] = prompt
            documents += [dict_line]
            if prompt not in dict_test:
            	dict_test[prompt] = 1
            else:
            	dict_test[prompt] += 1
    # print(dict_test)
    # print(len(dict_test))
    return documents

def entropy(e1, e2, rank, test_data, dict_rel):
	dict_bad = {}
	dd = {}
	e_l, rank_l = [], []
	for i in range(len(e1)):
		if e1[i]<0 and e2[i]>0:
			e_l += [e1[i]]
			rank_l += [rank[i]]
			prompt = test_data[i]['prompt']
			if "is located in" in prompt:
				dd[(test_data[i]["obj"], test_data[i]["sub"])] = 1
			if prompt not in dict_bad:
				if rank[i]<=10:
					dict_bad[prompt] = [1, 1]
				else:
					dict_bad[prompt] = [1, 0]
			else:
				dict_bad[prompt][0] += 1
				if rank[i]<=10:
					dict_bad[prompt][1] += 1

	print(len(dd))
	sort_dic = sorted(dict_bad.items(), key = lambda x: x[1][0], reverse = True)
	dict_b = {}
	all_b = 0
	for pair in sort_dic:
		# print(pair[0], pair[1], dict_rel[pair[0]])
		if dict_rel[pair[0]] not in dict_b:
			dict_b[dict_rel[pair[0]]] =pair[1][0]
		else:
			dict_b[dict_rel[pair[0]]] +=pair[1][0]
		all_b += pair[1][0]

	sort_dic = sorted(dict_b.items(), key = lambda x: x[1], reverse = True)
	for pair in sort_dic:
		print(pair[0], pair[1], pair[1]/all_b)

	plt.scatter(e_l, rank_l, s= 5, alpha = 0.8)

	plt.xlabel("Entropy change (Explicit)")
	plt.ylabel("Rank")
	plt.savefig("bad_e.png")
	plt.show()


def kl(kl1, kl2, rank, test_data, dict_rel):
	dict_bad = {}
	nb = 0
	rank_l = []
	kl_l = []
	for i in range(len(e1)):
		prompt = test_data[i]['prompt']
		if kl2[i]>0.5 and kl1[i]<0.2:
			kl_l += [kl1[i]]
			rank_l += [rank[i]]
			nb+=1
			if prompt not in dict_bad:
				if rank[i]<=10:
					dict_bad[prompt] = [1, 1]
				else:
					dict_bad[prompt] = [1, 0]
			else:
				dict_bad[prompt][0] += 1
				if rank[i]<=10:
					dict_bad[prompt][1] += 1

	sort_dic = sorted(dict_bad.items(), key = lambda x: x[1][0], reverse = True)
	for pair in sort_dic:
		print(pair)
	print(nb)
	plt.scatter(kl_l, rank_l, s= 5, alpha = 0.8)

	plt.xlabel("KL (Explicit)")
	plt.ylabel("Rank")
	plt.savefig("bad_kl.png")
	plt.show()
	# dict_b = {}
	# all_b = 0
	# for pair in sort_dic:
	# 	# print(pair[0], pair[1], dict_rel[pair[0]])
	# 	if dict_rel[pair[0]] not in dict_b:
	# 		dict_b[dict_rel[pair[0]]] =pair[1][0]
	# 	else:
	# 		dict_b[dict_rel[pair[0]]] +=pair[1][0]
	# 	all_b += pair[1][0]

	# sort_dic = sorted(dict_b.items(), key = lambda x: x[1], reverse = True)
	# for pair in sort_dic:
	# 	print(pair[0], pair[1], pair[1]/all_b)



e1= np.load('out/vanila_ent1.npy')
e2= np.load('out/vanila_ent2.npy')
kl1= np.load('out/vanila_kl1.npy')
kl2= np.load('out/vanila_kl2.npy')
rank= np.load('out/vanila_rank.npy')

e2_sub = np.load('out/vanila_ent2_sub.npy')
kl2_sub = np.load('out/vanila_kl2_sub.npy')


plt.scatter(e1, e2, s= 5, alpha = 0.8, label = "Explicit")
plt.scatter(e2_sub, e2, s= 5, alpha = 0.8, label = "Implicit-Sub")
plt.xlabel("Entropy Change")
plt.ylabel("Entropy Change")
plt.legend()
plt.savefig("sub_e.png")
plt.show()


plt.scatter(kl1, kl2, s= 5, alpha = 0.8, label = "Explicit")
plt.scatter(kl2_sub, kl2, s= 5, alpha = 0.8, label = "Implicit-Sub")
plt.xlabel("KL")
plt.ylabel("KL")
plt.legend()
plt.savefig("sub_kl.png")
plt.show()

print(P)



test_data = read_json('data/test.jsonl')
dict_rel = read_jj('relations.jsonl')
entropy(e1, e2, rank, test_data, dict_rel)
# kl(kl1, kl2, rank, test_data, dict_rel)

