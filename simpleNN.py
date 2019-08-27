import tensorflow as tf

class simpleNN:
	def __init__(self, num_user, num_item, dim_latent=32):
		
		# user id and item id 
		self.U = tf.placeholder(tf.int32, shape=[None, 1]) 
		self.I = tf.placeholder(tf.int32, shape=[None, 1])
		self.R = tf.placeholder(tf.float32, shape=[None, 1])

		# user embedding weights
		self.Wu = tf.Variable(tf.random_uniform([num_user, dim_latent], -1.0, 1.0), name="weight")
		# item embedding weights
		self.Wi = tf.Variable(tf.random_uniform([num_item, dim_latent], -1.0, 1.0), name="weight")

		# user, item embedding vectors for minibatch
		self.P = tf.nn.embedding_lookup(self.Wu, self.U)
		self.Q = tf.nn.embedding_lookup(self.Wi, self.I)

		self.pred = tf.reduce_sum(tf.multiply(self.P, self.Q), axis=2)
		self.loss = tf.reduce_mean(tf.square(tf.subtract(self.pred, self.R)))