import tensorflow as tf
import numpy as np


class RNN_Relation(object):
	def __init__(self, num_classes, word_dict_size, d1_dict_size, d2_dict_size, type_dict_size,pos_dict_size, sentMax, wv, molecule_embedding,molecule_emb_size=100,w_emb_size=200, d1_emb_size=10, d2_emb_size=10, type_emb_size=5,pos_emb_size=10, LSTM_hidden_unit=100, l2_reg_lambda = 0.0, pooling='max',learn_rate=0.001,batch_size=128):

		tf.reset_default_graph()
# 		emb_size = w_emb_size + d1_emb_size + d2_emb_size + type_emb_size
		emb_size = w_emb_size + d1_emb_size + d2_emb_size 		
#		emb_size = w_emb_size 

		self.sent_len = tf.placeholder(tf.int64, [None], name='sent_len')
		self.w  = tf.placeholder(tf.int32, [None, None], name="x")
		self.d1 = tf.placeholder(tf.int32, [None, None], name="x3")
		self.d2 = tf.placeholder(tf.int32, [None, None], name='x4')
		self.pos=tf.placeholder(tf.int32,[None,None],name='x5')

		self.mole1=tf.placeholder(tf.int32,[None,None])

		self.mole2=tf.placeholder(tf.int32,[None,None])

#		self.type = tf.placeholder(tf.int32, [None, None], name='x5')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Initialization
#		W_wemb =    tf.Variable(tf.random_uniform([word_dict_size, w_emb_size], -1.0, +1.0))
		with tf.variable_scope('embedding'):
			W_wemb  =   tf.Variable(wv)
			M_emb=tf.Variable(molecule_embedding)
			W_d1emb =   tf.Variable(tf.random_uniform([d1_dict_size, d1_emb_size],-(3/d1_emb_size)**0.5 , +(3/d1_emb_size)**0.5))
			W_d2emb =   tf.Variable(tf.random_uniform([d2_dict_size, d2_emb_size], -(3/d2_emb_size)**0.5, +(3/d2_emb_size)**0.5))
			#W_posemb=   tf.Variable(tf.random_uniform([pos_dict_size, pos_emb_size], -(3/pos_emb_size)**0.5, +(3/pos_emb_size)**0.5))
	#		W_typeemb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))
		
		# Embedding Layer
		

		emb0 = tf.nn.embedding_lookup(W_wemb, self.w)				#word embedding NxMx50
		emb3 = tf.nn.embedding_lookup(W_d1emb, self.d1)				#POS embedding  NxMx5
		emb4 = tf.nn.embedding_lookup(W_d2emb, self.d2)				#POS embedding  NxMx5
		molecule1_emb=tf.nn.embedding_lookup(M_emb, self.mole1)
		molecule2_emb=tf.nn.embedding_lookup(M_emb, self.mole2)
		molecule1_emb=tf.reshape(molecule1_emb,[-1,100])
		molecule2_emb=tf.reshape(molecule2_emb,[-1,100])
		#emb5=tf.nn.embedding_lookup(W_posemb, self.pos)


		self.X = tf.concat([emb0, emb3, emb4], 2 )
		#self.X = tf.concat([emb0, emb3, emb4,emb5], 2 )
		#self.cnn_step(sentMax)
		#self.X=tf.concat([self.X,self.X1],2)




		
		#Recurrent Layer
		cell_f = tf.contrib.rnn.LSTMCell(num_units=LSTM_hidden_unit, state_is_tuple=True)
		cell_b = tf.contrib.rnn.LSTMCell(num_units=LSTM_hidden_unit, state_is_tuple=True)
		outputs, states = tf.nn.bidirectional_dynamic_rnn(
									cell_fw	=cell_f, 
									cell_bw	=cell_b, 
									dtype	=tf.float32, 	
									sequence_length=self.sent_len, 
									inputs	=self.X
								)

		output_fw, output_bw = outputs						#NxMx100
		states_fw, states_bw = states
		#print 'output_fw', output_fw.get_shape()
	
		h = tf.concat([output_fw, output_bw], 2)				#NxMx200
		#print 'h', h.get_shape()

		#Attention Layer		
		
		
		h = tf.expand_dims(h, -1)						#NxMx200x1
		#print 'h', h.get_shape()
		
		m = tf.reduce_max(self.sent_len)
		if pooling == 'max':
		   pooled = tf.nn.max_pool(h, ksize=[1, sentMax, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")	#Nx1x200x1
		else:
		   pooled = tf.reduce_sum(h, 1)
#		pooled = tf.nn.avg_pool(h, ksize=[1, sentMax, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")	#Nx1x200x1
		#print 'pooled', pooled.get_shape()
		
		h2 = tf.reshape(pooled, [-1, 2*LSTM_hidden_unit])				#?x200
		#print 'h2', h2.get_shape()
				
		# dropout layer	 
		h2 = tf.nn.dropout(h2, self.dropout_keep_prob)
		h2 = tf.tanh(h2)

		h2=tf.concat([h2,molecule1_emb,molecule2_emb], -1)					



		W = tf.get_variable(name="W",shape=[2*LSTM_hidden_unit+200, num_classes],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)#随机生成W的初始矩阵

		b = tf.get_variable(name="b",shape=[num_classes],initializer=tf.zeros_initializer(),dtype=tf.float32)#随机生成b的偏移矩阵
		scores = tf.nn.xw_plus_b(h2, W, b, name="scores")#全连接层			#200x8
		#print 'score', scores.get_shape()

		self.predictions = tf.argmax(scores, 1, name="predictions")
		tv=tf.trainable_variables()
		# for v in tv :
		# 	if v.name.split('/')[0]!='embedding':
		# 		print(v.name)
		regularization_cost=tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if v.name.split('/')[0]!='embedding'])
		#losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
		self.loss = tf.reduce_mean(losses)  + l2_reg_lambda * regularization_cost/(2*batch_size)

		self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		#self.optimizer = tf.train.AdamOptimizer(learn_rate)
		self.optimizer = tf.train.RMSPropOptimizer(learn_rate)


		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
		# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
		# session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
	

	def test_step(self,sess, W_batch, Sent_len, d1_batch, d2_batch, t_batch,pos_batch, y_batch,drop_out):

#		w,d1,d2,typet = paddData([W_batch, d1_batch, d2_batch, t_batch])
		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.pos:pos_batch,
#				self.type	:t_batch,
				self.sent_len 	:Sent_len,
				self.dropout_keep_prob: drop_out,
				self.input_y 	:y_batch
					}
		step, loss, accuracy, predictions = sess.run([self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)

			#print "Accuracy in test data", accuracy
		return predictions, accuracy

	def train_step(self,sess, W_batch, Sent_len, d1_batch, d2_batch,mole1_batch,mole2_batch, t_batch,pos_batch, y_batch, drop_out):
		#Padding data 
		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.mole1:mole1_batch,
				self.mole2:mole2_batch,
				#self.pos:pos_batch,
#				self.type	:t_batch,
				self.sent_len 	:Sent_len,
				self.dropout_keep_prob: drop_out,
				self.input_y 	:y_batch
					}
		_, step, loss, accuracy, predictions = sess.run([self.train_op, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
			#print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
		return loss

	def cnn_step(self,sentMax):
		shape=tf.shape(self.X)
		self.filter_num=40
		self.dim=220
		#print(shape[0],shape[1],shape[2])
		
		filter_shape1=[1,self.dim,self.filter_num]
		filter_shape2=[2,self.dim,self.filter_num]
		filter_shape3=[3,self.dim,self.filter_num]
		filter_shape4=[4,self.dim,self.filter_num]
		filter_shape5=[5,self.dim,self.filter_num]		    



		W_conv1 = tf.get_variable(shape=filter_shape1,initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32,trainable=True ,name="W_conv1")#W为卷积核
		W_conv2 = tf.get_variable(shape=filter_shape2,initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32,trainable=True ,name="W_conv2")#W为卷积核
		W_conv3 = tf.get_variable(shape=filter_shape3,initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32,trainable=True ,name="W_conv3")#W为卷积核
		W_conv4 = tf.get_variable(shape=filter_shape4,initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32,trainable=True ,name="W_conv4")#W为卷积核
		W_conv5 = tf.get_variable(shape=filter_shape5,initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32,trainable=True ,name="W_conv5")#W为卷积核
		
		
		b1=tf.get_variable(shape=[self.filter_num],initializer=tf.zeros_initializer(),trainable=True ,name="b1")
		b2=tf.get_variable(shape=[self.filter_num],initializer=tf.zeros_initializer(),trainable=True ,name="b2")
		b3=tf.get_variable(shape=[self.filter_num],initializer=tf.zeros_initializer(),trainable=True ,name="b3")
		b4=tf.get_variable(shape=[self.filter_num],initializer=tf.zeros_initializer(),trainable=True ,name="b4")
		b5=tf.get_variable(shape=[self.filter_num],initializer=tf.zeros_initializer(),trainable=True ,name="b5")
		


		conv1=tf.nn.conv1d(self.X, W_conv1, stride=1,padding= "SAME",name="conv1")
		conv2=tf.nn.conv1d(self.X, W_conv2, stride=1,padding= "SAME",name="conv2")
		conv3=tf.nn.conv1d(self.X, W_conv3, stride=1,padding= "SAME",name="conv3")
		conv4=tf.nn.conv1d(self.X, W_conv4, stride=1,padding= "SAME",name="conv4")
		conv5=tf.nn.conv1d(self.X, W_conv5, stride=1,padding= "SAME",name="conv5")
		#填充一维以最大池化
		conv1=tf.nn.bias_add(conv1, b1,name="addbias1")
		conv2=tf.nn.bias_add(conv2, b2,name="addbias2")
		conv3=tf.nn.bias_add(conv3, b3,name="addbias3")
		conv4=tf.nn.bias_add(conv4, b4,name="addbias4")
		conv5=tf.nn.bias_add(conv5, b5,name="addbias5")

		# a=tf.reshape(conv1,[shape[0],shape[1],100])
		# print(conv1.shape)


		# conv1=tf.expand_dims(conv1,axis=-1)
		# conv2=tf.expand_dims(conv2,axis=-1)
		# conv3=tf.expand_dims(conv3,axis=-1)
		# conv4=tf.expand_dims(conv4,axis=-1)
		# conv5=tf.expand_dims(conv5,axis=-1)
		
		

		# pooled1=tf.nn.max_pool(conv1,ksize=[1,1,1,1] , strides=[1,sentMax,1,1],padding="SAME",name="pooled1")
		# pooled2=tf.nn.max_pool(conv2,ksize=[1,1,1,1] , strides=[1,sentMax,1,1],padding="SAME",name="pooled2")
		# pooled3=tf.nn.max_pool(conv3,ksize=[1,1,1,1] , strides=[1,sentMax,1,1],padding="SAME",name="pooled3")
		# pooled4=tf.nn.max_pool(conv4,ksize=[1,1,1,1] , strides=[1,sentMax,1,1],padding="SAME",name="pooled4")
		# pooled5=tf.nn.max_pool(conv5,ksize=[1,1,1,1] , strides=[1,sentMax,1,1],padding="SAME",name="pooled5")
		

		# stroke_input1=tf.reshape(pooled1, [shape[0],shape[1],self.filter_num])
		# stroke_input2=tf.reshape(pooled2, [shape[0],shape[1],self.filter_num])
		# stroke_input3=tf.reshape(pooled3, [shape[0],shape[1],self.filter_num])
		# stroke_input4=tf.reshape(pooled4, [shape[0],shape[1],self.filter_num])
		# stroke_input5=tf.reshape(pooled5, [shape[0],shape[1],self.filter_num])


		#self.X=tf.concat([stroke_input1,stroke_input2,stroke_input3,stroke_input4,stroke_input5], axis=-1)
		self.X1=tf.concat([conv1,conv2,conv3,conv4,conv5], axis=-1)
		





