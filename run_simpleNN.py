import pickle
import tensorflow as tf
import numpy as np
from random import *
import os
from simpleNN import simpleNN
import argparse

# tensorflow session environment setting
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# parse arguments
parser = argparse.ArgumentParser() # ArgumentParser 객체 생성
parser.add_argument("--dim_latent", type=int, default=32, help="latent dimension")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--n_epoch", type=int, default=100, help="batch size")
parser.add_argument("--train", type=int, default=1, help="train flag")
parser.add_argument("--debug", type=int, default=0, help="debug flag")
args = parser.parse_args()

# preprocessing
PREPRO_DIR = './data/amazon/preprocessed'
SAVE_DIR = './save'
SAVE_FILE = os.path.join(SAVE_DIR, 'model')
with open(os.path.join(PREPRO_DIR, 'amazon.para'),'rb') as f:
	para = pickle.load(f) 
with open(os.path.join(PREPRO_DIR, 'amazon.train'),'rb') as f:
	train = pickle.load(f) 
with open(os.path.join(PREPRO_DIR, 'amazon.test'),'rb') as f:
	test = pickle.load(f) 
train = np.array(train)
test = np.array(test)
num_user = para['num_user']
num_item = para['num_item']
dim_latent = args.dim_latent
batch_size = args.batch_size
n_epoch = args.n_epoch

""" load graph"""
model = simpleNN(num_user, num_item, dim_latent)
""" optimizer """
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(model.loss)

# declare Saver object in order to save tensorflow graph 
saver = tf.train.Saver()

if __name__ == '__main__':
	
	""" training phase """
	if args.train == True:
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(n_epoch):
				loss_epoch = 0
				num_step = int(len(train) / batch_size)
				for batch_num in range(num_step):
					# build minibatch 
					start_index = batch_num * batch_size
					end_index = min((batch_num + 1) * batch_size, len(train))
					batches_train = train[start_index: end_index]
					uid = batches_train[:,0].astype('int32') 
					iid = batches_train[:,1].astype('int32')
					ratings =batches_train[:,2].astype('float32')
					batches = {model.U:uid, model.I:iid, model.R:ratings}
					loss_step, _, = sess.run([model.loss, train_op], feed_dict=batches)
					loss_epoch += loss_step
				loss_epoch = loss_epoch/num_step

				if epoch % 10 == 0:
					""" test phase """
					loss_epoch_test = 0
					num_step = int(len(test) / batch_size)
					for batch_num in range(num_step):
						# build minibatch 
						start_index = batch_num * batch_size
						end_index = min((batch_num + 1) * batch_size, len(test))
						batches_test = test[start_index: end_index]
						uid = batches_test[:,0].astype('int32') 
						iid = batches_test[:,1].astype('int32')
						ratings =batches_test[:,2].astype('float32')
						batches = {model.U:uid, model.I:iid, model.R:ratings}
						loss_step = sess.run(model.loss, feed_dict=batches)
						loss_epoch_test += loss_step

					loss_epoch_test = loss_epoch_test/num_step
					print("epoch:", epoch, "\t |test:", loss_epoch_test, "\t |train:", loss_epoch)

			# save computational tensorflow graph 
			print("Learning finished, saved")
			saver.save(sess, SAVE_FILE, global_step=epoch+1)

	if args.debug == True:
		with tf.Session() as sess:
			b_test = int(input('select minibatch size for test'))
			num_model = input('select model epoch for test')
			shuffle_indices = np.random.permutation(np.arange(len(test)))
			shuffled_test = test[shuffle_indices]
			start = randrange(len(test)-b_test)

			""" load model test """
			# pick a random batch
			batches_test = test[start:start+b_test]
			uid = batches_test[:,0].astype('int32') 
			iid = batches_test[:,1].astype('int32')
			ratings =batches_test[:,2].astype('float32')
			saver.restore(sess, os.path.join(SAVE_DIR, 'model-{}'.format(num_model)))
			print("rating={}, prediction={}".format(ratings, 
													model.pred.eval(feed_dict={model.U:uid ,
																			   model.I:iid})))