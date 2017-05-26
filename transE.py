#coding:utf-8
from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import datetime
import ctypes
import matplotlib.pyplot as plt

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")

class Config(object):

	def __init__(self):
		self.L1_flag = True
		self.hidden_size = 100
		self.nbatches = 100
		self.entity = 0
		self.relation = 0
		self.trainTimes = 3000
		self.margin = 1.0

class TransEModel(object):

	def __init__(self, config):

		entity_total = config.entity
		relation_total = config.relation
		batch_size = config.batch_size
		size = config.hidden_size
		margin = config.margin

		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])

		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

		with tf.name_scope("embedding"):
			self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
			pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
			pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
			neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
			neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
			neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

		if config.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
		else:
			pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
			neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)			

		with tf.name_scope("output"):
			self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

		#For each triple in test data,predict score for all possible relations using the entities of that test triple.
		self.prediction=tf.reduce_sum(abs(pos_h_e+pos_r_e-pos_t_e),1,keep_dims=True)

def main(_):
	lib.init()
	config = Config()
	config.relation = lib.getRelationTotal()
	config.entity = lib.getEntityTotal()
	config.batch_size = lib.getTripleTotal() / config.nbatches

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer(uniform = False)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				trainModel = TransEModel(config = config)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.GradientDescentOptimizer(0.001)
			grads_and_vars = optimizer.compute_gradients(trainModel.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			saver = tf.train.Saver()
			sess.run(tf.initialize_all_variables())

			def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
					trainModel.neg_h: neg_h_batch,
					trainModel.neg_t: neg_t_batch,
					trainModel.neg_r: neg_r_batch
				}
				_, step, loss = sess.run(
					[train_op, global_step, trainModel.loss], feed_dict)
	 			return loss

	 		#Predict function to predict scores for test data
	 		def predict(test_h,test_t,test_r):
	 			feed_dict={
	 				trainModel.pos_h:test_h,
	 				trainModel.pos_t:test_t,
	 				trainModel.pos_r:test_r
	 			}

	 			phe,result=sess.run([trainModel.pos_h,trainModel.prediction],feed_dict)
	 			return result

	 		ph = np.zeros(config.batch_size, dtype = np.int32)
	 		pt = np.zeros(config.batch_size, dtype = np.int32)
	 		pr = np.zeros(config.batch_size, dtype = np.int32)
	 		nh = np.zeros(config.batch_size, dtype = np.int32)
	 		nt = np.zeros(config.batch_size, dtype = np.int32)
	 		nr = np.zeros(config.batch_size, dtype = np.int32)

	 		ph_addr = ph.__array_interface__['data'][0]
	 		pt_addr = pt.__array_interface__['data'][0]
	 		pr_addr = pr.__array_interface__['data'][0]
	 		nh_addr = nh.__array_interface__['data'][0]
	 		nt_addr = nt.__array_interface__['data'][0]
	 		nr_addr = nr.__array_interface__['data'][0]

	 		loss=np.zeros((config.trainTimes,1))
	 		for times in range(config.trainTimes):
	 			res = 0.0
				for batch in range(config.nbatches):
					lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
					#print ph.shape,pt.shape,pr.shape
					res += train_step(ph, pt, pr, nh, nt, nr)
					current_step = tf.train.global_step(sess, global_step)
				loss[times,:]=res
				print times
				print res
			plt.plot(range(config.trainTimes),loss)
			plt.show()
			saver.save(sess, 'model_full_train.vec')	
			saver.restore(sess,'./model_full_train.vec')
			print "Model saved!"
			# np.savetxt('Entity_Embed.txt',sess.run(trainModel.ent_embeddings))
			# np.savetxt('Rel_Embed.txt',sess.run(trainModel.rel_embeddings))

			# res=predict([8867],[4230],[565])
			# print "Probability of given link is:",res	
			relation_array=np.array(range(0,1345,1))
			test_data=pd.read_csv('triple2id.txt',sep="\t",header=None)
			hits1=0
			hits5=0
			hits10=0
			for i in range(test_data.shape[0]):
				e1=test_data.iloc[i][0]
				e2=test_data.iloc[i][1]
				r=test_data.iloc[i][2]
				#Run a for loop for all the relations,get them into a single numpy array or list. Do that outisde here so it can be used
				#multiple times inside. Probably just run a loop and keep on appending! 
				results=predict(np.array([e1]*1345),np.array([e2]*1345),relation_array)
				results_with_id=np.hstack((np.reshape(relation_array,[relation_array.shape[0],1]),results))
				results_with_id=results_with_id[np.argsort(results_with_id[:,1])]
				results_with_id=results_with_id[:,0]
				# results_with_id=results_with_id[::-1]
				# temp=results_with_id[:,1]
				# temp=temp[::-1]
				# temp=results_with_id[:,1]
				# sigs=1.0/(1.0+np.exp(-1.0*temp))
				# print temp
				# print sigs[0:10]
				# print results_with_id[0:100]
				hit_1=results_with_id[0]
				hit_5=results_with_id[0:5]
				hit_10=results_with_id[0:10]
				if np.in1d(r,hit_1):
					hits1+=1
				if np.in1d(r,hit_5):
					hits5+=1
				if np.in1d(r,hit_10):
					hits10+=1

			mean_hit1=hits1/test_data.shape[0]
			mean_hit5=hits5/test_data.shape[0]
			mean_hit10=hits10/test_data.shape[0]
			print "Mean of Hits@1:",mean_hit1
			print "Mean of Hits@5:",mean_hit5
			print "Mean of Hits@10:",mean_hit10

if __name__ == "__main__":
	tf.app.run()

