//기본 함수 불러오기
import tensorflow as tf

//편의를 위해 데이터를 자동으로 다운로드하고 설치하는 코드
from tensorflow.examples.tutorials.mnist import input_data

//데이터를 가져옴
# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

//각 이미지들은 784차원의 벡터이므로, [None, 784] 형태의 부정소숫점으로 이루어진 2차원 텐서로 표현. None 은 가변가능표시 
# Set up model
x = tf.placeholder(tf.float32, [None, 784])
//Variable초기값 설정. 784차원 이미지 벡터를 곱하여 10차원 벡터의 증거를 만듬.
W = tf.Variable(tf.zeros([784, 10]))
//b는 y = softmax(Wx + b)에서의 b. 출력에 더할 수 있음.
b = tf.Variable(tf.zeros([10]))
//x가 여러입력으로 구성된 2D 텐서일 경우를 다룰 수 있게 하기 위해 x와 W의 위치를 바꿈.
// 그리고 거기에 b를 더해 y = softmax(Wx + b)를 완성.
y = tf.nn.softmax(tf.matmul(x, W) + b)

//교차 엔트로피를 구현하기 위해 정답을 입력하기 위한 placeholder.
y_ = tf.placeholder(tf.float32, [None, 10])

//교차엔트로피값 -시그마(y'log(y)) 계산. * 한개의 교차엔트로피값이 아닌 넣은 이미지 전체에 대한 교차엔트로피의 합
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
//학습도를 0.01로 준 경사하강법 알고리즘(변수들의 비용을 줄임)을 통해 교차엔트로피를 수정하도록 함.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

//만든 변수들의 초기화
# Session
init = tf.initialize_all_variables()

//세션에서 모델 시작하고 변수들 초기화
sess = tf.Session()
sess.run(init)


/1000번 가르치기
# Learning
for i in range(1000):
  //학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옴
  batch_xs, batch_ys = mnist.train.next_batch(100)
  //피딩을 실행
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

//예측이 실제와 맞았는지를 검증. 얼마나 많이 맞았는지 확인하는 것. 부울값으로 리턴. 평균값으로 비율 계산.
# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
//테스트 데이터 대상으로 정확도 확인.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

//정확도값 출력하기. 위 예제에서는 약 91%가 나와야한다.
# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))