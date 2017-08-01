import tensorflow as tf
import numpy as np
import matplotlib as plt
import pickle

# prameters
alpha = 0.05
learning_rate = 0.05
train_file = '../data/training.pk1'
test_file = '../data/sample.pk1'
class_type = 100
# prameter end

with open(train_file, 'rb') as f:
    X_data = pickle.load(f)
    y_data = pickle.load(f)
with open(test_file, 'rb') as f:
    X_test = pickle.load(f)

X_tmp = []
theta_tmp = []
for each in X_data:
    X_tmp.append(X_data[1] + X_data[2] * 10000 + X_data[3] * 1000)
    theta_tmp.append(X_data[0])

# initialize
X = np.zeros(70000)
theta = np.zeros(6000)
# random init
#
pairs = tf.placeholder(tf.int32, [None, 2])
x = tf.Variable(tf.truncated_normal([70000, class_type]))
theta = tf.Variable(tf.truncated_normal([6000, class_type]))
bias = tf.bias_variable(tf.s)
def cost_func():
    n = pairs.shape[0]
    s = tf.zeros([1])
    for i in range(n):
        idx = pairs[i][0]
        idt = pairs[i][1]
        s = s + tf.matmul(x[idx], theta[idt].T)
    return s
y_ = cost_func()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
