# -*- coding: utf-8 -*-

"""
CIFAR-100 Convolutional Neural Networks(CNN) 예제

next_batch function is copied from edo's answer

https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
"""

import tensorflow as tf
import numpy as np
import os

# CIFAR-100 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar100 import load_data

# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch(num, data, labels):
  '''
  `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
  '''
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# CNN 모델을 정의합니다. 
def build_CNN_classifier(x):
  # 입력 이미지
  x_image = x

  # VGG11 (A)
  # Conv3-64
  xavier_initializer = tf.contrib.layers.xavier_initializer()
  W_conv1 = tf.Variable(xavier_initializer(shape=[3, 3, 3, 64]))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # maxpool
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob) 

  # conv3-128
  W_conv2 = tf.Variable(xavier_initializer(shape=[3, 3, 64, 128]))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1_drop, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # maxpool
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob) 

  # conv3-256
  W_conv3 = tf.Variable(xavier_initializer(shape=[3, 3, 128, 256]))
  b_conv3 = tf.Variable(tf.constant(0.1, shape=[256]))
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2_drop, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

  # conv3-256
  W_conv4 = tf.Variable(xavier_initializer(shape=[3, 3, 256, 256]))
  b_conv4 = tf.Variable(tf.constant(0.1, shape=[256]))
  h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

  # maxpool
  h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob) 

  # conv3-512
  W_conv5 = tf.Variable(xavier_initializer(shape=[3, 3, 256, 512]))
  b_conv5 = tf.Variable(tf.constant(0.1, shape=[512])) 
  h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4_drop, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

  # conv3-512
  W_conv6 = tf.Variable(xavier_initializer(shape=[3, 3, 512, 512]))
  b_conv6 = tf.Variable(tf.constant(0.1, shape=[512])) 
  h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6)

  # maxpool
  h_pool6 = tf.nn.max_pool(h_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool6_drop = tf.nn.dropout(h_pool6, keep_prob) 

  # conv3-512
  W_conv7 = tf.Variable(xavier_initializer(shape=[3, 3, 512, 512]))
  b_conv7 = tf.Variable(tf.constant(0.1, shape=[512])) 
  h_conv7 = tf.nn.relu(tf.nn.conv2d(h_pool6_drop, W_conv7, strides=[1, 1, 1, 1], padding='SAME') + b_conv7)

  W_conv8 = tf.Variable(xavier_initializer(shape=[3, 3, 512, 512]))
  b_conv8 = tf.Variable(tf.constant(0.1, shape=[512])) 
  h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME') + b_conv8)

  # maxpool
  h_pool8 = tf.nn.max_pool(h_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_pool8_drop = tf.nn.dropout(h_pool8, keep_prob)
  h_pool8_flat = tf.reshape(h_pool8_drop, [-1, 1*1*512])

  # FC-4096
  W_fc9 = tf.Variable(xavier_initializer(shape=[1 * 1 * 512, 4096]))
  b_fc9 = tf.Variable(tf.constant(0.1, shape=[4096]))
  h_fc9 = tf.nn.relu(tf.matmul(h_pool8_flat, W_fc9) + b_fc9)
  h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob) 
 
 # FC-4096
  W_fc10 = tf.Variable(xavier_initializer(shape=[4096, 4096]))
  b_fc10 = tf.Variable(tf.constant(0.1, shape=[4096]))
  h_fc10 = tf.nn.relu(tf.matmul(h_fc9_drop, W_fc10) + b_fc10)
  h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob) 

  # FC-1000
  W_fc11 = tf.Variable(xavier_initializer(shape=[4096, 100]))
  b_fc11 = tf.Variable(tf.constant(0.1, shape=[100]))
  logits = tf.matmul(h_fc10_drop, W_fc11) + b_fc11
  y_pred = tf.nn.softmax(logits)

  return y_pred, logits

# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 100])
keep_prob = tf.placeholder(tf.float32)

# CIFAR-100 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()
# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 100),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 100),axis=1)

# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
y_pred, logits = build_CNN_classifier(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
  # 모든 변수들을 초기화한다. 
  sess.run(tf.global_variables_initializer())

  # 만약 저장된 모델과 파라미터가 있으면 이를 불러옵니다. (Restore)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print(ckpt.model_checkpoint_path)
  
  num_step = 40001
  # num_step만큼 최적화를 수행합니다.
  for i in range(num_step):
    batch = next_batch(256, x_train, y_train_one_hot.eval())

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))

    # 1000 Step마다 validation 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 1000 == 0:
      test_accuracy = 0.0  
      for j in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
      test_accuracy = test_accuracy / 10;
      print("테스트 데이터 정확도: %f" % test_accuracy)

    # 10000 Step마다 tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
    if i % 10000 == 0:      
      saver.save(sess, checkpoint_path, global_step=i)

    # 50% 확률의 Dropout을 이용해서 학습을 진행합니다.
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

  # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.  
  test_accuracy = 0.0  
  for i in range(10):
    test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
    test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
  test_accuracy = test_accuracy / 10;
  print("테스트 데이터 정확도: %f" % test_accuracy) # Accuracy : 약 50%