# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np

# set data path
train_path = "iris_training.csv"
test_path = "iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data(train_path, test_path, y_name='Species'):
  """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""

  train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
  train_x, train_y = train, train.pop(y_name)

  test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
  test_x, test_y = test, test.pop(y_name)

  return (train_x, train_y), (test_x, test_y)

# load data
(train_x, train_y), (test_x, test_y) = load_data(train_path, test_path)

# 학습을 위한 설정값들을 정의합니다.
learning_rate = 0.001
num_epochs = 100
display_step = 1    # 손실함수 출력 주기
input_size = 4 # SepalLength, SepalWidth, PetalLength, PetalWidth
hidden1_size = 256
hidden2_size = 256
output_size = 3 # Setosa, Versicolor, Virginica

# 입력값과 출력값을 받기 위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, output_size])

# ANN 모델을 정의합니다.
def build_ANN(x):
  W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
  b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
  H1_output = tf.nn.relu(tf.matmul(x,W1) + b1)
  W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
  b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
  H2_output = tf.nn.relu(tf.matmul(H1_output,W2) + b2)
  W_output = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]))
  b_output = tf.Variable(tf.random_normal(shape=[output_size]))
  logits = tf.matmul(H2_output,W_output) + b_output

  return logits

# ANN 모델을 선언합니다.
predicted_value = build_ANN(x)

# 손실함수와 옵티마이저를 정의합니다.
# tf.nn.softmax_cross_entropy_with_logits 함수를 이용하여 활성함수를 적용하지 않은 output layer의 결과값(logits)에 softmax 함수를 적용합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 세션을 열고 그래프를 실행합니다.
with tf.Session() as sess:
  # 변수들에 초기값을 할당합니다.
  sess.run(tf.global_variables_initializer())

  # 지정된 횟수만큼 최적화를 수행합니다.
  for epoch in range(num_epochs):
    average_loss = 0.

    batch_x = train_x.values
    batch_y = sess.run(tf.one_hot(train_y.values, 3))

    # 옵티마이저를 실행해서 파라마터들을 업데이트합니다.
    _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
    # 평균 손실을 측정합니다.
    average_loss += current_loss / batch_x.shape[0]
    # 지정된 epoch마다 학습결과를 출력합니다.
    if epoch % display_step == 0:
      print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch+1), average_loss))

  # 테스트 데이터를 이용해서 학습된 모델이 얼마나 정확한지 정확도를 출력합니다.
  correct_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  print("정확도(Accuracy): %f" % (accuracy.eval(feed_dict={x: test_x.values, y: sess.run(tf.one_hot(test_y.values, 3))}))) # 정확도: 96%(100epoch)