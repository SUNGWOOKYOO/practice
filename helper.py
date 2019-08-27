import re 
import numpy as np
import tensorflow as tf
# https://docs.python.org/3.6/library/itertools.html
import itertools
# https://docs.python.org/3.6/library/collections.html#collections.Counter 
from collections import Counter
from progress.bar import Bar

""" helper function for preprocessing """

def clean_str(string):
	""" 
	Tokenize sentences in string 
	"""
	string = re.sub(r"[^A-Za-z]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n'\t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def pad_sentences(review_words, words_length, padding_word="<PAD/>"):
	"""
	Pads all words to the same length. 
	words_length is defined by the benchmark length of long words.
	Returns padding words.
	"""
	padding_review_words = {} 
	for key, words in review_words.items():
		# 어떤 user 또는 item에 대한 review에서 사용된 words 수가 기준 길이 보다 작을 경우
		if words_length > len(words):
			# padding을 통해 일정하게 만든다(CNN의 input으로 만들기 위해)
			num_padding = words_length - len(words)
			# pad word를 뒤에 list형식으로 append 한다.
			new_words = words + [padding_word] * num_padding
			padding_review_words[key] = new_words
		# 기준길이보다 크거나 같을 경우 
		else:
			# benchmark의 길이인 words_length 만큼만 사용
			new_words = words[:words_length]
			padding_review_words[key] = new_words

	return padding_review_words

def build_voca(words_list):
	"""
	Builds a vocabulary dictionary based on words for each user or item.
	Returns vocabulary dictionay and it's inverse dictionary.
	"""
	# Build vocabulary
	# itertools.chain(*words_list): 모든 words 내의 단어를 이은다.
	# Counter 객체로 각 단어별 frequency 를 파악한다.
	word_counts = Counter(itertools.chain(*words_list))
	# Mapping from index to word
	# word_counts.most_common(): word의 freq가 높은 순으로 tuple list 생성 
	# voca_inv: 단어들을 lexicographical order로 정렬
	voca_inv = [x[0] for x in word_counts.most_common()]
	voca_inv = list(sorted(voca_inv))
	# Mapping from word to index
	# vocabulary에 indexing 을 함
	# 이때, '\<PAD>'의 index=0
	voca = {x: i for i,x in enumerate(voca_inv)}

	return [voca, voca_inv]

def word2vocaid(review_words, voca):
	"""
	Maps sentences and labels to vectors based on a vocabulary
	"""
	# user text review
	# words[key]의 value를 [voca[key]]로 변경
	words2 = {}
	for key, words in review_words.items():
		word_id = np.array([voca[word] for word in words])
		words2[key] = word_id

	return words2

""" helper function for training """

def build_batches(minibatch, user_words, item_words, Tu, Ti):
	"""
	input: minibatch shape=[None, 3, 1], dtype=dnarray  
	return uid, iid, ratings, Widuser, Widitem
	"""
	uid = minibatch[:,0].astype('int32') # [None, 1]
	iid = minibatch[:,1].astype('int32') # [None, 1]
	ratings = minibatch[:,2].astype('float32') # [None, 1]

	Widuser = np.array(user_words[uid[0][0]].reshape(-1, Tu)) # [None, Tu]
	i = 0
	for u in uid:
		if i > 0:
			Widuser = np.append(Widuser, user_words[u[0]].reshape(-1, Tu), axis=0)
		i += 1
	Widitem = np.array(item_words[iid[0][0]].reshape(-1, Ti)) # [None, Ti]
	j = 0
	for p in iid:
		if j > 0:
			Widitem = np.append(Widitem, item_words[p[0]].reshape(-1, Ti), axis=0)
		j += 1
		
	return uid, iid, ratings, Widuser, Widitem 

def get_pretrained_embedding(vocab, glove, embedding_dim):
	w = 0
	initW = np.random.uniform(-1.0, 1.0, (len(vocab), embedding_dim))
	bar = Bar('preprocess glove file', max=len(vocab)) # visualization
	for word in vocab.keys():
		if word in glove.keys():
			idx = vocab[word]
			initW[idx] = glove[word]
			w += 1
		bar.next()
	bar.finish()
	print("number of pre-trained words", w)
	return initW


""" helper function for computing mutual scores """
def get_fscore_matrix(U, V, lu, li, f):
	"""
	U.shape=[b, lu, 1, f]
	V.shape=[b, li, 1, f]
	
	return A # shape=[b,lu,li]
	"""
	cu = tf.reshape(U, shape=[-1, lu, f]) # [b, lu, f]
	ci = tf.reshape(V, shape=[-1, li, f]) # [b, li, f]
	# squared norms of each row in A and B
	na = tf.reduce_sum(tf.square(cu), 2) # [b, lu]
	nb = tf.reduce_sum(tf.square(ci), 2) # [b, li]

	# na as a row and nb as a column vectors
	na = tf.expand_dims(na, axis=-1) # [n=b, i=lu, j=1]
	nb = tf.expand_dims(nb, axis=1) # [n=b, i=1, lj=i]

	mid = tf.einsum('nik,njk->nij', cu, ci) # [b, lu, li]

	D = tf.sqrt(na - 2*mid + nb)
	A = tf.divide(1, tf.add(D, 1))
	return A

if __name__ == '__main__':
	print('This is helper module')