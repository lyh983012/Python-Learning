import tensorflow as tf
import cv2
import numpy as np
import time

data = np.load('MNIST.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

while True:
  index_show = np.random.randint(0, 60000)
  sampleX = trainX[index_show,...]
  sampleY = trainY[index_show]
  cv2.imshow('image', cv2.resize(sampleX,(560,560),interpolation=cv2.INTER_NEAREST))
  print(np.argmax(sampleY))
  if cv2.waitKey(-1) & 0xFF == ord('q'):
    break

sess = tf.InteractiveSession()


with tf.name_scope('input'):
  x_image = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
  x = x_image/255.

with tf.name_scope('label'):
  label = tf.placeholder(shape=[None, 10], dtype=tf.float32)

with tf.name_scope('conv1'):
  W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.3))

  h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID'))
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('conv2'):
  W_conv2 = tf.Variable(tf.truncated_normal(([5, 5, 32, 64]), stddev=0.05))

  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID'))
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('reshape'):
  h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])

with tf.name_scope('fullconnect1'):
  W_fc1 = tf.Variable(tf.truncated_normal(([4 * 4 * 64, 512]), stddev=0.04))
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1))

with tf.name_scope('output'):
  W_fc2 = tf.Variable(tf.truncated_normal(([512, 10]), stddev=0.06))
  y = tf.matmul(h_fc1, W_fc2)

with tf.name_scope('cross_entropy'):
  loss = tf.losses.softmax_cross_entropy(onehot_labels=label,logits=y)

with tf.name_scope('train'):
  optimizer = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)

with tf.name_scope('accuracy'):
  error = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(y, axis=1), tf.argmax(label, axis=1)), tf.float32))


sess.run(tf.global_variables_initializer())

print('Optimization Start!')
# Training cycle
for epoch in range(100):
  print('Epoch: %03d' % (epoch + 1), end=' ')
  loss_sum = 0.
  error_sum = 0.
  t0 = time.time()
  for i in range(600):
    batchX = trainX[(i * 100):((i + 1) * 100), ...]
    batchY = trainY[(i * 100):((i + 1) * 100), ...]

    [_, loss_delta, error_delta] = sess.run([optimizer, loss, error], feed_dict={x_image: batchX, label: batchY})
    loss_sum += loss_delta
    error_sum += error_delta

  print('Loss: %.6f Train: %.4f' % (loss_sum/600, error_sum/600), end=' ')
  FPS = 60000 / (time.time() - t0)

  errorTest = error.eval({x_image: testX, label: testY})
  print('Test: %.4f FPS: %d' % (errorTest, FPS))





