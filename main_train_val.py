from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import sklearn as sk
import random
import csv
import re
import collections
import pickle
import sys
import os
sys.path.append("source")
from utils import *

from rnn_train import *
#from rnn_train import *
#from cnn_train import *
#from att_rnn import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


test_train=""

embSize = 200
d1_emb_size=10
d2_emb_size=10
type_emb_size=10
pos_emb_size=10
numfilter = 100#LSTM的维度，

#out_file = sys.argv[1]
#sent_out = sys.argv[2]


sent_out = 'results/multi_sents_'

num_epochs = 100
#N = 4
check_point = [4,7,10,13,17]
batch_size=64
reg_para =0.0001#正则化
drop_out =1#无dropout
learn_rate=0.001

# ftrain = "dataset/ddi/neg_filtered/train_data95.txt"
# fval = "dataset/ddi/neg_filtered/train_data05.txt"
# ftest = "dataset/ddi/neg_filtered/test_data.txt"

ftrain = "hierarchical_DDI2013_preprocess/molecule_dataset/dataset1/small_molecule_dataset_train.txt"
fval = "hierarchical_DDI2013_preprocess/molecule_dataset/dataset1/small_molecule_dataset_test.txt"
ftest = "hierarchical_DDI2013_preprocess/molecule_dataset/dataset1/small_molecule_dataset_test.txt"


# ftrain = "hierarchical_DDI2013_preprocess/train_processed2.txt"
# fval = "hierarchical_DDI2013_preprocess/test_processed2.txt"
# ftest = "hierarchical_DDI2013_preprocess/test_processed2.txt"

ftrain_pos="hierarchical_DDI2013_preprocess/POS/train_pos.txt"
ftest_pos="hierarchical_DDI2013_preprocess/POS/test_pos.txt"
#wefile = "/home/sunil/embeddings/cbow_300d_gvkcorpus.txt"
wefile = "word_embeddings/embedding.txt"

molecule_embedding_file="molecule_embedding/needed_embedding4.txt"


time_stamp=str(int(time.time()))
model_save_path="saved_models/"+time_stamp
out_file = model_save_path+"/detail_results.txt"
if not os.path.exists(model_save_path): os.makedirs(model_save_path)

Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables = dataRead(ftrain)#
Tr_word_list, Tr_d1_list, Tr_d2_list, Tr_type_list,Tr_drug1_list,Tr_drug2_list = makeFeatures(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list)#Tr_d1_list为距离特征

V_sent_contents, V_entity1_list, V_entity2_list, V_sent_lables = dataRead(fval)
V_word_list, V_d1_list, V_d2_list, V_type_list,V_drug1_list,V_drug2_list = makeFeatures(V_sent_contents, V_entity1_list, V_entity2_list)

Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables = dataRead(ftest)
Te_word_list, Te_d1_list, Te_d2_list, Te_type_list,Te_drug1_list,Te_drug2_list = makeFeatures(Te_sent_contents, Te_entity1_list, Te_entity2_list)

Tr_sent_pos_list=posRead(ftrain_pos)
Te_sent_pos_list=posRead(ftest_pos)

print ("train_size", len(Tr_word_list))
#print "val_size", len(V_word_list)
print ("test_size", len(Te_word_list))
print("molecule sentence size ",len(Tr_drug1_list), len(Tr_drug2_list))

train_sent_lengths, val_sent_lengths, test_sent_lengths = findSentLengths([Tr_word_list, V_word_list, Te_word_list])
sentMax = max(train_sent_lengths + val_sent_lengths + test_sent_lengths)

print ("max sent length", sentMax)#最大句子长度

train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
val_sent_lengths = np.array(test_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')


label_dict = {'false':0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
#label_dict = {'false':0, 'true':1}

word_dict = makeWordList([Tr_word_list, V_word_list, Te_word_list])

molecule_dict=makeMoleculeList(molecule_embedding_file)

d1_dict = makeDistanceList([Tr_d1_list, V_d1_list, Te_d1_list])
d2_dict = makeDistanceList([Tr_d2_list, V_d2_list, Te_d2_list])

type_dict = makeDistanceList([Tr_type_list, V_type_list, Te_type_list])

pos_dict=makePOSList([Tr_sent_pos_list,Te_sent_pos_list])

print ("word dictonary length", len(word_dict))

# Word Embedding
wv = readWordEmb(word_dict, wefile, embSize)
molecule_embedding=readMoleculeEmb(molecule_dict,molecule_embedding_file, 100)		

# Mapping Train
W_train =   mapWordToId(Tr_word_list, word_dict)
d1_train = mapWordToId(Tr_d1_list, d1_dict)
d2_train = mapWordToId(Tr_d2_list, d2_dict)


molecule1_train=mapDrugToId(Tr_drug1_list, molecule_dict)
molecule2_train=mapDrugToId(Tr_drug2_list, molecule_dict)


T_train = mapWordToId(Tr_type_list,type_dict)
POS_train=mapWordToId(Tr_sent_pos_list,pos_dict)

Y_t = mapLabelToId(Tr_sent_lables, label_dict)
#print(Y_t)

Y_train = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_train[i][Y_t[i]] = 1.0

#Y_train为最后的标签矩阵

#Mapping Validation
W_val =   mapWordToId(V_word_list, word_dict)


molecule1_val=mapDrugToId(V_drug1_list, molecule_dict)
molecule2_val=mapDrugToId(V_drug2_list, molecule_dict)


d1_val = mapWordToId(V_d1_list, d1_dict)
d2_val = mapWordToId(V_d2_list, d2_dict)
T_val = mapWordToId(V_type_list,type_dict)

Y_t = mapLabelToId(V_sent_lables, label_dict)
Y_val = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_val[i][Y_t[i]] = 1.0

# Mapping Test
W_test =   mapWordToId(Te_word_list, word_dict)

molecule1_test=mapDrugToId(Te_drug1_list, molecule_dict)
molecule2_test=mapDrugToId(Te_drug2_list, molecule_dict)

d1_test = mapWordToId(Te_d1_list, d1_dict)
d2_test = mapWordToId(Te_d2_list, d2_dict)
T_test = mapWordToId(Te_type_list, type_dict)
POS_test=mapWordToId(Te_sent_pos_list,pos_dict)
Y_t = mapLabelToId(Te_sent_lables, label_dict)
Y_test = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_test[i][Y_t[i]] = 1.0

#padding


W_train, d1_train, d2_train, T_train,POS_train, W_val, d1_val, d2_val, T_val, W_test, d1_test, d2_test, T_test,POS_test = paddData([W_train, d1_train, d2_train, T_train,POS_train, W_val, d1_val, d2_val, T_val, W_test, d1_test, d2_test, T_test,POS_test], sentMax) 

print ("train", len(W_train))
print ("test", len(W_test))


with open('train_test_rnn_data.pickle', 'wb') as handle:
	pickle.dump(W_train, handle)
	pickle.dump(d1_train, handle)
	pickle.dump(d2_train, handle)
	pickle.dump(T_train, handle)
	pickle.dump(Y_train, handle)
	pickle.dump(train_sent_lengths, handle)

	pickle.dump(W_val, handle)
	pickle.dump(d1_val, handle)
	pickle.dump(d2_val, handle)
	pickle.dump(T_val, handle)
	pickle.dump(Y_val, handle)
	pickle.dump(val_sent_lengths, handle)

	pickle.dump(W_test, handle)
	pickle.dump(d1_test, handle)
	pickle.dump(d2_test, handle)
	pickle.dump(T_test, handle)
	pickle.dump(Y_test, handle)
	pickle.dump(test_sent_lengths, handle)

	pickle.dump(wv, handle)
	pickle.dump(word_dict, handle)
	pickle.dump(d1_dict, handle)
	pickle.dump(d2_dict, handle)
	pickle.dump(type_dict, handle)	 
	pickle.dump(label_dict, handle) 
	pickle.dump(sentMax, handle)



#vocabulary size
word_dict_size = len(word_dict)
d1_dict_size = len(d1_dict)
d2_dict_size = len(d2_dict)
type_dict_size = len(type_dict)
pos_dict_size=len(pos_dict)
label_dict_size = len(label_dict)

rev_word_dict = makeWordListReverst(word_dict)
rev_pos_dict = makeWordListReverst(pos_dict)
rev_label_dict = {0:'false', 1:'advise', 2:'mechanism', 3:'effect', 4:'int'}

fp = open(out_file, 'a+')	# keep precision recall
#fsent = open(sent_out, 'w') 	# keep sentence and its results
def test_step(sess,W, sent_lengths, d1, d2, T,POS, Y,drop_out):
	n = len(W)

	ra = int(n/batch_size)
	#int(train_len/batch_size) + 1
	samples = []
	for i in range(ra):
		samples.append(range(batch_size*i, batch_size*(i+1)))
	samples.append(range(batch_size*(i+1), n))

	acc = [] 
	pred = []
	for i in samples:
		p,a = rnn.test_step(sess,W[i], sent_lengths[i], d1[i], d2[i], T[i],POS[i], Y[i],drop_out)
		
		pred.extend(p)

	return pred
#print 'drop_out, reg_rate', drop_out, reg_para
if test_train=='test':
	ckpt_file=tf.train.latest_checkpoint("saved_models/L20_dropout1_leanrate0.001_RMSopt/")

	rnn = RNN_Relation(label_dict_size, 		# output layer size
				word_dict_size, 		# word embedding size
				d1_dict_size, 			# position embedding size	
				d2_dict_size, 			# position embedding size
				type_dict_size, 		# type emb. size
				sentMax, 			# length of sentence
				wv,	# word embedding
				molecule_embedding,

				molecule_emb_size=100,
				learn_rate=learn_rate,			
				d1_emb_size=d1_emb_size, 	# emb. length
				d2_emb_size=d2_emb_size, 	
				type_emb_size=type_emb_size,	
				LSTM_hidden_unit=numfilter, 		# number of hidden nodes in RNN
				w_emb_size=embSize, 		# dim. word emb
				l2_reg_lambda=reg_para,# l2 reg
				batch_size=batch_size		
			)
	saver=tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, ckpt_file)
		y_pred_test  = test_step(sess,W_train, train_sent_lengths, d1_train, d2_train, T_train, Y_train)
		y_true_test = np.argmax(Y_train, 1)
		# y_pred_test  = test_step(sess,W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test)
		# y_true_test = np.argmax(Y_test, 1)





		f1_scorea=f1_score(y_true_test, y_pred_test, [1,2,3,4], average='micro')
		print("    ",f1_scorea,"!!!!!!!!")
		exit()


rnn = RNN_Relation(label_dict_size, 		# output layer size
			word_dict_size, 		# word embedding size
			d1_dict_size, 			# position embedding size	
			d2_dict_size, 			# position embedding size
			type_dict_size, 		# type emb. size
			pos_dict_size,
			sentMax, 			# length of sentence
			wv,	# word embedding
			molecule_embedding,
			molecule_emb_size=100,
			learn_rate=learn_rate,			
			d1_emb_size=d1_emb_size, 	# emb. length
			d2_emb_size=d2_emb_size, 	
			type_emb_size=type_emb_size,
			pos_emb_size=pos_emb_size,	
			LSTM_hidden_unit=numfilter, 		# number of hidden nodes in RNN
			w_emb_size=embSize, 		# dim. word emb
			l2_reg_lambda=reg_para,		# l2 reg
			batch_size=batch_size
		)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)  
sess.run(tf.global_variables_initializer())	

train_len = len(W_train)#训练的语句

loss_list = []

test_res = []
val_res = []

fscore_val = []
fscore_test = []



f=open(model_save_path+'/results.txt','a+')
f.writelines("batch_size:"+str(batch_size)+"    learn_rate:"+str(learn_rate)+"     L2:"+str(reg_para)+"    drop_out:"+str(drop_out)+"\n")
f.close()

# print(molecule1_train,type(molecule1_train))
# print(d1_train,type(d1_train))
# exit()

num_batches_per_epoch = int(train_len/batch_size) + 1
iii = 0		#Check point number
high_f1=0
f1_list=[]
for epoch in range(num_epochs):	
	shuffle_indices = np.random.permutation(np.arange(train_len))
	W_tr =  W_train[shuffle_indices]
	d1_tr = d1_train[shuffle_indices]
	d2_tr = d2_train[shuffle_indices]
	M1_tr=molecule1_train[shuffle_indices]
	M2_tr=molecule2_train[shuffle_indices]
	T_tr = T_train[shuffle_indices]
	POS_tr=POS_train[shuffle_indices]
	Y_tr = Y_train[shuffle_indices]
	S_tr = train_sent_lengths[shuffle_indices]
	loss_epoch = 0.0




	for batch_num in range(num_batches_per_epoch):


		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, train_len)
		loss = rnn.train_step(sess,W_tr[start_index:end_index], S_tr[start_index:end_index], d1_tr[start_index:end_index], 
			d2_tr[start_index:end_index],M1_tr[start_index:end_index],M2_tr[start_index:end_index], T_tr[start_index:end_index],POS_tr[start_index:end_index], Y_tr[start_index:end_index], drop_out)
		loss_epoch += loss
		print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time()))," epoch:",epoch+1," batch_num:",batch_num," total_batch:",num_batches_per_epoch," loss:",loss)

	print("loss_epoch:",loss_epoch)

	loss_list.append(round(loss_epoch, 5) ) 

	iii += 1

	saver = tf.train.Saver()
	path = saver.save(sess, model_save_path+'/model_'+str(iii)+'.ckpt')

	y_pred_test  = test_step(sess,W_test, test_sent_lengths, d1_test, d2_test, T_test,POS_test, Y_test,drop_out)

	y_true_test = np.argmax(Y_test, 1)
	f1_scorea=f1_score(y_true_test, y_pred_test, [1,2,3,4], average='micro')
	f1_list.append(f1_scorea)
	if f1_scorea>high_f1:
		high_f1=f1_scorea
	print("the highest f1 score is:",high_f1)
	class1_f1=str(f1_score(y_true_test, y_pred_test, [1], average='micro' ))
	class2_f1=str(f1_score(y_true_test, y_pred_test, [2], average='micro' ))
	class3_f1=str(f1_score(y_true_test, y_pred_test, [3], average='micro' ))
	class4_f1=str(f1_score(y_true_test, y_pred_test, [4], average='micro' ))

	print("epoch:",epoch+1,'f1_scorea:',f1_scorea)
	print("		class1:",class1_f1)
	print("		class2:",class2_f1)
	print("		class3:",class3_f1)
	print("		class4:",class4_f1)
	f=open(model_save_path+'/results.txt','a+')
	f.write("epoch:"+str(epoch+1)+" f1_score:"+str(f1_scorea)+" loss:"+str(loss_epoch)+'\n')
	f.write("		class1:"+class1_f1+'\n')
	f.write("		class2:"+class2_f1+'\n')
	f.write("		class3:"+class3_f1+'\n')
	f.write("		class4:"+class4_f1+'\n')
	f.close()
	

	
	fscore_test.append( f1_scorea)
	test_res.append([y_true_test, y_pred_test])

if test_train!="test":
	plot_x=[]
	for i in range(1,101):
		plot_x.append(i)
	plt.plot(plot_x,f1_list)
	plt.xlabel("epoch")
	plt.ylabel("F1-score")
	
	plt.savefig(model_save_path+'/line.jpg')
	plt.show()

sess.close()










