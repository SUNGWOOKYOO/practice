from platform import python_version
import os
import json
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split # version 0.21.3
from helper import clean_str, pad_sentences, build_voca, word2vocaid
"""
clean_str(): Tokenize sentences in string 
pad_sentences(): Pads all words to the same length. 
word2vocaid(): Maps sentences and labels to vectors based on a vocabulary
"""

if __name__ == '__main__':
	print('python={}'.format(python_version()))
	print('pandas==', end='')
	print(pd.__version__)
	print('numpy==', end='')
	print(np.__version__)
	
	DATA_NAME = 'Musical_Instruments_5.json'
	# DATA_NAME = 'Office_Products_5.json'
	DATA_DIR = './data/amazon'
	DATA_JSON = os.path.join(DATA_DIR, DATA_NAME)
	PREPRO_DIR = './data/amazon/preprocessed'

	""" read data from *.json """
	users_id = []
	items_id = []
	ratings = []
	reviews = []
	with open(DATA_JSON,'r') as f:
		for line in f:
			js = json.loads(line)
			if str(js['reviewerID']) == 'unknown':
				print("unknown")
				continue
			if str(js['asin']) == "unknown":
				print("unknown")
				continue
			reviews.append(js['reviewText'])
			users_id.append(str(js['reviewerID']) + ",")
			items_id.append(str(js['asin']) + ",")
			ratings.append(str(js['overall']))
	
	df = pd.DataFrame(
    {'user_id': pd.Series(users_id),
     'item_id': pd.Series(items_id),
     'ratings': pd.Series(ratings),
     'reviews': pd.Series(reviews)}
    )[['user_id', 'item_id', 'ratings', 'reviews']]
	
	"""
	https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html 
	user_id column을 pandas.Series.astype 중에서 dtype=category 데이터 타입으로 바꾼후
	"""
	user_id_int = df.user_id.astype('category').cat.codes
	item_id_int = df.item_id.astype('category').cat.codes
	df.user_id = user_id_int.astype('int64')
	df.item_id = item_id_int.astype('int64')
	df.ratings = df.ratings.astype('float64')
	
	""" split train, valid, test pandas.DataFrame w/o reviews """ 
	dfs = df[['user_id', 'item_id', 'ratings']]
	df_train, df_test = train_test_split(dfs, test_size=0.2, random_state=0, shuffle=True)
	df_test, df_valid = train_test_split(df_test, test_size=0.5, random_state=0, shuffle=True)
	
	""" [exploration] explore the sparsity """ 
	# 이 부분은 굳이 하지 않아도 num_user, num_item을 다른 방식으로 구할 수 있다. 
	rating_table = dfs.pivot_table(index='user_id', columns='item_id', values='ratings')
	num_user, num_item = rating_table.shape
	# distinct # of (user, item)  
	print("num_user={}, num_item={}".format(num_user, num_item))
	# density ratio for ratings 
	print("density ratio={0:.3f}%".format((1 - rating_table.isnull().sum().sum()/(num_user*num_item))*100))
	
	""" preprocess review sentence corpus """  
	user_reviews = {} # 유저가 리뷰를 쓴 문장들 모음 
	item_reviews = {}
	# user_rid, item_rid는 NARRE에서는 쓰임(여기서 안쓰임)
	user_rid = {} # 유저가 어느 아이템에 review를 썻는가의 정보 모음
	item_rid = {}

	for i in df.loc[df_train.index].values:
		user_id = i[0]
		item_id = i[1]
		review_text = i[3]
		if user_id in user_reviews:
			user_reviews[user_id].append(review_text)
			user_rid[user_id].append(item_id)
		else:
			user_reviews[user_id] = [review_text]
			user_rid[user_id] = [item_id]
		if item_id in item_reviews:
			item_reviews[item_id].append(review_text)
			item_rid[item_id].append(user_id)
		else:
			item_reviews[item_id] = [review_text]
			item_rid[item_id] = [user_id]

	""" 
	padding in test, valid set
	test 와 valid set 을 숨기면서 id에 대한 value값을 '0'으로 padding 해 놓는다.
	"""
	for i in df.loc[pd.concat([df_test, df_valid]).index].values:
		user_id = i[0]
		item_id = i[1]
		review_text = i[3]

		# user_review[user_id]가 train dataset에 없는경우엔
		if user_id not in user_reviews:
			# print(user_id) # 예를 찾기 위한 debug
			# test dataset user_id를 user_reviews key list 에 추가한다.(나중에 prediction 할때 쓰이므로)
			# 또한, 그때의 value는 padding을 의미하는 '0'을 넣는다. 
			user_rid[user_id] = [0]
			user_reviews[user_id] = ['0']

		# user_review[user_id]가 train dataset에 있었어서 review가 이미 저장되어있다면 
		else: None
			# print(user_id) # 예를 찾기위한 debug 
			# test dataset의 review 는 저장하지 않는다.
		# user의 경우와 동일하므로 생략
		if item_id not in item_reviews:
			item_reviews[item_id] = [0]
			item_rid[item_id] = ['0']
			
	""" tokenize review and create train, valid, test dataset for training phase  """
	# load_data(...)
	uid_train = []
	iid_train = []
	y_train = []
	u_words = {}
	i_words = {}
	for line in df_train.values:
		user_id = int(line[0])
		item_id = int(line[1])

		uid_train.append(user_id)
		iid_train.append(item_id)

		# user_id in u_words 가 True 인 경우는 user_review에 정보를 모두 앞서 모아놨으므로 처리 안해도 됨
		if user_id not in u_words:
			u_words[user_id] = ''
			for s in user_reviews[user_id]:
				u_words[user_id] = u_words[user_id] + " " + s.strip()
			# clean_str을 사용하여 review sentences 를 tokenize 한다.
			u_words[user_id] = clean_str(u_words[user_id])
			u_words[user_id] = u_words[user_id].split(" ")

		if item_id not in i_words:
			i_words[item_id] = ''
			for s in item_reviews[item_id]:
				i_words[item_id] = i_words[item_id] + " " + s.strip()
			i_words[item_id] = clean_str(i_words[item_id])
			i_words[item_id] = i_words[item_id].split(" ")
		y_train.append(float(line[2]))

	uid_valid = []
	iid_valid = []
	y_valid = []
	for line in df_valid.values:
		user_id = int(line[0])
		item_id = int(line[1])

		uid_valid.append(user_id)
		iid_valid.append(item_id)

		# train에서 없는 review 정보이지만, valid 에서 사용해야하므로 pad값을 넣음  
		if user_id not in u_words:
			# print(user_id) # 예를 찾기 위한 debug
			u_words[user_id] = '<PAD/>'
			# u_words[user_id] = clean_str(u_words[user_id])
			u_words[user_id] = u_words[user_id].split(" ")

		if item_id not in i_words:
			# print(item_id) # 예를 찾기 위한 debug
			i_words[item_id] = '<PAD/>'
			# i_words[item_id] = clean_str(i_word[item_id])
			i_words[item_id] = i_words[item_id].split(" ")

		y_valid.append(float(line[2]))
		
	uid_test = []
	iid_test = []
	y_test = []
	for line in df_test.values:
		user_id = int(line[0])
		item_id = int(line[1])

		uid_test.append(user_id)
		iid_test.append(item_id)

		# train에서 없는 review 정보이지만, test에서 사용해야하므로 pad값을 넣음  
		if user_id not in u_words:
			# print(user_id) # 예를 찾기 위한 debug
			u_words[user_id] = '<PAD/>'
			# u_words[user_id] = clean_str(u_words[user_id])
			u_words[user_id] = u_words[user_id].split(" ")

		if item_id not in i_words:
			# print(item_id) # 예를 찾기 위한 debug
			i_words[item_id] = '<PAD/>'
			# i_words[item_id] = clean_str(i_word[item_id])
			i_words[item_id] = i_words[item_id].split(" ")

		y_test.append(float(line[2]))
		

	# 각 유저가 쓴 review의 words 길이 
	words_length_list = np.array([len(x) for x in u_words.values()])
	# benchmark로 특정 길이인 words_length 를 정한다.(85% 정도의 상대적으로 긴 words)
	u_words_len = np.sort(words_length_list)[int(0.85 * num_user) - 1]

	words_length_list = np.array([len(x) for x in i_words.values()])
	i_words_len = np.sort(words_length_list)[int(0.85 * num_item) - 1]

	print("u_words_len={}".format(u_words_len))
	print("i_words_len={}".format(i_words_len))
	
	# padding 작업
	u_words = pad_sentences(u_words, u_words_len)
	i_words = pad_sentences(i_words, i_words_len)
	# 2D list shape = (num_*, *_len) for *_words
	u_words_list = [x for x in u_words.values()]
	i_words_list = [x for x in i_words.values()]
	# voca_user: {word: word_id}
	voca_user, voca_inv_user = build_voca(u_words_list)
	voca_item, voca_inv_item = build_voca(i_words_list)
	# words를 voca_id로 바꾸는 작업('\<PAD>'의 voca_id = 0)
	u_words = word2vocaid(u_words, voca_user)
	i_words = word2vocaid(i_words, voca_item)

	num_voca_user = len(voca_user)
	num_voca_item = len(voca_item)
	print("num_voca_user={}".format(num_voca_user))
	print("num_voca_item={}".format(num_voca_item))
	
	# numpy.array 자료형으로 형변환
	y_train = np.array(y_train)
	y_valid = np.array(y_valid)
	y_test = np.array(y_test)
	uid_train = np.array(uid_train)
	uid_valid = np.array(uid_valid)
	uid_test = np.array(uid_test)
	iid_train = np.array(iid_train)
	iid_valid = np.array(iid_valid)
	iid_test = np.array(iid_test)

	# random으로 record 순서를 섞고 numpy 자료형으로 만들며 shape를 알맞게 조정
	np.random.seed(2019)
	shuffle_indices = np.random.permutation(np.arange(len(y_train)))
	userid_train = uid_train[shuffle_indices]
	itemid_train = iid_train[shuffle_indices]
	y_train = y_train[shuffle_indices]
	y_train = y_train[:, np.newaxis]
	y_valid = y_valid[:, np.newaxis]
	y_test = y_test[:, np.newaxis]

	userid_train = userid_train[:, np.newaxis]
	itemid_train = itemid_train[:, np.newaxis]
	userid_valid = uid_valid[:, np.newaxis]
	itemid_valid = iid_valid[:, np.newaxis]
	userid_test = uid_test[:, np.newaxis]
	itemid_test = iid_test[:, np.newaxis]

	# training phase 에서 사용하기 위해서 batch 형식으로 변환
	# [(np.array([user]), np.array([item]), np.array([ratings]))] 형식으로 변환됨
	batches_train = list(zip(userid_train, itemid_train, y_train))
	batches_valid = list(zip(userid_valid, itemid_valid, y_valid))
	batches_test = list(zip(userid_test, itemid_test, y_test))
	
	with open(os.path.join(PREPRO_DIR, 'amazon.train'),'wb') as f:
		pickle.dump(batches_train, f)

	with open(os.path.join(PREPRO_DIR, 'amazon.valid'),'wb') as f:
		pickle.dump(batches_valid, f)
		
	with open(os.path.join(PREPRO_DIR, 'amazon.test'),'wb') as f:
		pickle.dump(batches_test, f)

	para = {}
	para['num_user'] = num_user
	para['num_item'] = num_item
	para['voca_user'] = voca_user
	para['voca_item'] = voca_item
	para['user_words'] = u_words
	para['item_words'] = i_words

	with open(os.path.join(PREPRO_DIR, 'amazon.para'),'wb') as f:
		pickle.dump(para, f)
		
	print("PREPRO_DIR={}\n amazon.train, amazon.valid, amazon.test, amazon.para file saved ".format(PREPRO_DIR))
	print('preprocessing end ==============================')
    