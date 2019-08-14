# from nltk.tokenize import WordPunctTokenizer
# tokenizer = WordPunctTokenizer()
# def findSentLengths(tr_te_list):
# 	lis = []
# 	for lists in tr_te_list:
# 		lis.append([len(l) for l in lists])
# 	return lis
# def preProcess(sent):
# # 	sent = sent.lower()
# # 	sent = sent.replace('/',' ')

# # #	sent = sent.replace('(','')
# # #	sent = sent.replace(')','')
# # #	sent = sent.replace('[','')
# # #	sent = sent.replace(']','')
# # 	sent = sent.replace('.','')
# # #	sent = sent.replace(',',' ')
# # #	sent = sent.replace(':','')
# # #	sent = sent.replace(';','')
	
# # 	sent = tokenizer.tokenize(sent)
# # 	sent = ' '.join(sent)
# # 	#sent = re.sub('\d', 'dg',sent)
# # 	sent = re.sub('([0-9]{1,}[.][0-9]*)', "dga", sent)#浮点数字替换为dga
# # 	sent = re.sub('([0-9]{1,})', "dgb", sent)#数字替换为dgb
# 	sent = sent.lower()
# 	sent = tokenizer.tokenize(sent)
# 	sent = ' '.join(sent)

# 	return sent


# fname="hierarchical_DDI2013_preprocess/test_data_cut_processd2.txt"
# #print (Input File Reading)
# fp = open(fname, 'r')
# samples = fp.read().strip().split('\n\n')
# sent_lengths   = []		#1-d array
# sent_contents  = []		#2-d array [[w1,w2,....] ...]
# sent_lables    = []		#1-d array
# entity1_list   = []		#2-d array [[e1,e1_t] [e1,e1_t]...]
# entity2_list   = []		#2-d array [[e1,e1_t] [e1,e1_t]...]
# for sample in samples:
# 	#print(sample)
# 	#print(1)
# 	sent, entities, relation = sample.strip().split('\n')
# #		if len(sent.split()) > 100:
# #			continue
# 	e1, e1_t, e2, e2_t = entities.split('\t') 
# 	sent_contents.append(sent.lower())
# 	entity1_list.append([e1, e1_t])
# 	entity2_list.append([e2, e2_t])
# 	sent_lables.append(relation)

# # print(sent_contents)
# # print(1)
# # print(entity1_list)
# # print(2)
# # print(sent_lables)
# # print(3)
# sent_list=sent_contents
# word_list = []
# d1_list = []
# d2_list = []
# type_list = []
# if len(sent_list)!=len(entity1_list):
# 	print("list wrong")
# 	exit()
# for sent, ent1, ent2 in zip(sent_list, entity1_list, entity2_list):
# 	sent = preProcess(sent)
# #		print sent
# 	sent_list1 = sent.split()
	
# 	# entity1 = preProcess(ent1[0]).split()
# 	# entity2 = preProcess(ent2[0]).split()
# 	s1 = sent_list1.index('druga')
# 	s2 = sent_list1.index('drugb') 
# 	# distance1 feature	
# 	d1 = []
# 	for i in range(len(sent_list1)):
# 		if i < s1 :
# 			d1.append(str(i - s1))
# 		elif i > s1 :
# 			d1.append(str(i - s1 ))
# 		else:
# 			d1.append('0')
# 	#distance2 feature		
# 	d2 = []
# 	for i in range(len(sent_list1)):
# 		if i < s2:
# 			d2.append(str(i - s2))
# 		elif i > s2:
# 			d2.append(str(i - s2))
# 		else:
# 			d2.append('0')
# 	#type feature
# 	t = []
# 	for i in range(len(sent_list1)):
# 		t.append('Out')
# 	t[s1] = ent1[1]		
# 	t[s2] = ent2[1]

# 	word_list.append(sent_list1)
# 	d1_list.append(d1)
# 	d2_list.append(d2)
# 	type_list.append(t) 

# train_sent_lengths, val_sent_lengths, test_sent_lengths = findSentLengths([word_list, word_list, word_list])
# sentMax = max(train_sent_lengths + val_sent_lengths + test_sent_lengths)

# print(train_sent_lengths,sentMax)
# i,j=0,0
# a=450000*12
# b=450000*13
# w=open("word_embeddings/embed13.txt","w")
# with open('word_embeddings/wikipedia-pubmed-and-PMC-w2v.txt','r') as f:
# 	for line in f:
# 		i+=1
# 		if i<=b and i>a:
# 			line=line.strip().split('	')
# 			word_=line[0]
# 			vec_=line[1].split(',')
# 			w.writelines(word_+"	"+" ".join(vec_)+"\n")
# 			if i==b:
# 				print("finish")
# 				w.close()
# 				exit()			
			

# 	print("finally finish")
# 	w.close(
from utils import *
def readWordEmb(word_list, fname, embSize=200):
	print ("Reading word vectors")
	word_em=open('word_embeddings/embedding.txt','a')

	count = 0
	with open(fname, 'r') as f:
		for line in f :

			vs = line.strip().split("	")
			if vs[0] in word_list:
				count+=1
				vect=vs[1].split()
				word_em.writelines(vs[0]+"	"+" ".join(vect)+"\n")

	word_em.close()

	print ("number of known word in word embedding", count)
	#return wordemb

ftrain = "hierarchical_DDI2013_preprocess/train_processed2.txt"
fval = "hierarchical_DDI2013_preprocess/test_processed2.txt"
ftest = "hierarchical_DDI2013_preprocess/test_processed2.txt"
Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables = dataRead(ftrain)#
Tr_word_list, Tr_d1_list, Tr_d2_list, Tr_type_list = makeFeatures(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list)#Tr_d1_list为距离特征

V_sent_contents, V_entity1_list, V_entity2_list, V_sent_lables = dataRead(fval)
V_word_list, V_d1_list, V_d2_list, V_type_list = makeFeatures(V_sent_contents, V_entity1_list, V_entity2_list)

Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables = dataRead(ftest)
Te_word_list, Te_d1_list, Te_d2_list, Te_type_list = makeFeatures(Te_sent_contents, Te_entity1_list, Te_entity2_list)
word_dict = makeWordList([Tr_word_list, V_word_list, Te_word_list])


readWordEmb(word_dict, "word_embeddings/embed13.txt")