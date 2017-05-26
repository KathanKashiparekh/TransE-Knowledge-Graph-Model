import numpy as np 
import tensorflow as tf  

sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,'./model_full_train.vec')
