import numpy as np
from task import Task
import tensorflow as tf

learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.999

img_pre_size = 2048
txt_pre_size = 50
embed_size = 128
batch_size = 2


#define input
img_pre = tf.placeholder(tf.float32, shape=(batch_size, img_pre_size))
txt_pre = tf.placeholder(tf.float32, shape=(batch_size, txt_pre_size))
y = tf.placeholder(tf.float32, shape=(batch_size,))

#forward pass img and txt
w_img = tf.Variable(tf.random_normal([img_pre_size, embed_size], stddev=0.001))
b_img = tf.Variable(tf.constant(0.0, shape=(embed_size,)))
w_txt = tf.Variable(tf.random_normal([txt_pre_size, embed_size], stddev=0.01))
b_txt = tf.Variable(tf.constant(0.0, shape=(embed_size,)))

img = tf.nn.relu(tf.matmul(img_pre, w_img) + b_img)#We could use a conv net here.
txt = tf.nn.relu(tf.matmul(txt_pre, w_txt) + b_txt)#We might use two layers for each type|

#loss
losses = tf.reduce_mean(np.square(img - txt), axis=1)
loss = tf.reduce_mean(tf.maximum(tf.multiply(losses, y), 0))
optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
minimize = optimizer.minimize(loss)

tr_loss, dev_loss, dev_acc = {}, {}, {}
task = Task(batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in itertools.count():
        img_inpt, txt_inpt, ys = task.train_batch()
        _, tr_loss[i] = sess.run([minimize, loss], feed_dict={img_pre: img_inpt, txt_pre: txt_inpt, y: ys})
        print ('Tr loss: ', tr_loss[i])
