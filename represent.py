import numpy as np
from task import Task
import tensorflow as tf

learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.999

vocab_size = 10000
embed_size = 128
batch_size = 100


mobilenet = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2")
img_h, img_w = hub.get_expected_image_size(module)
channels = 3


#define input
img_pre = tf.placeholder(tf.string, shape=(batch_size, img_w, img_h, channels))
txt_pre = tf.placeholder(tf.float32, shape=(batch_size))

#embedding matrix
embedding_mat = tf.Variable(
	tf.random_normal((vocab_size, embed_size), stddev=.1))

def encode_image(images):
	with tf.variable_scope("encode_image"):
		mobilenet_vector = mobilenet(images)
		image_encoding = tf.layers.dense(
			mobilenet_vector, embed_size, activation=None)
		return image_encoding

def embed_text(texts):
	words = tf.string_split(texts)
	hash_ids = tf.SparseTensor(
		indices=words.indices,
		values=tf.strings.to_hash_bucket_fast(words.values),
		dense_shape=words.dense_shape
    )
    embeddings = tf.nn.embedding_lookup_sparse(embedding_mat, hash_ids)
	return embeddings

def encode_text(embeddings):
	with tf.variable_scope("encode_text"):
		hidden = tf.layers.dense(
			embeddings, embed_size, activation=tf.nn.relu)
		encoding = tf.layers.dense(hidden, embed_size, activation=tf.nn.relu)
		return encoding

#forward pass img and txt
text_embeddings = embed_text(txt_pre)
text_encodings = encode_text(text_embeddings)
image_encodings = encode_image(img_pre)

#loss
similarity = tf.matmul(
	text_encodings, image_encodings, transpose_b=True)
target_similarity = tf.eye(batch_size)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
	labels=target_similarity, logits=similarity)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
minimize = optimizer.minimize(loss)
