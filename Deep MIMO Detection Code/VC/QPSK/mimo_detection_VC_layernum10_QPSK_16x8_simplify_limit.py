import tensorflow as tf
import numpy as np
import scipy.io
from numpy import linalg as la

LOGDIR = "path/layernum10/16x8/5000-simplify-residual-limit7"
K = 8  # Number of users
N = 16 # Number of receiving antenna
modType = 4  # bpsk:2 qpsk:4 16qam:16
modName = 'qpsk'
batch_size = 5000  # Define the batch size
STEPS = 30000  # Number of iteration
# Hdata = scipy.io.loadmat('Hn1_16x8_7.3962')
# H = Hdata['Hn1']
# Hpart1 = np.concatenate((np.real(H), np.imag(H)))
# Hpart2 = np.concatenate((-np.imag(H), np.real(H)))
# realH = np.concatenate((Hpart1, Hpart2), axis=1)
# realH = realH.astype('float32')
# modulation
table_bpsk = np.array([1, -1])  # modType=2
table_qpsk = np.array(
    [-0.707107 + 0.707107j, -0.707107 - 0.707107j, 0.707107 + 0.707107j, 0.707107 - 0.707107j])  # modType=4
modTable = {'bpsk': table_bpsk, 'qpsk': table_qpsk}


def modulation(sourceSeq, num_Symbol, mod_Type, mod_Name):
    mod = modTable[mod_Name]
    if mod_Type == 2:
        mod_Seq = np.zeros((num_Symbol, 1))
    else:
        mod_Seq = np.zeros((num_Symbol, 1), dtype='complex')
    if mod_Type == 2:
        for i in range(num_Symbol):
            index = sourceSeq[i][0]
            mod_Seq[i] = mod[index]
            i = i + 1

    if mod_Type == 4:
        for i in range(num_Symbol):
            index = sourceSeq[i * 2][0] * 2 + sourceSeq[i * 2 + 1][0]
            mod_Seq[i] = mod[index]
            i = i + 1
    return mod_Seq


# demodulation
def demodulation(receiveSeq, num_Symbol, mod_Type):
    if mod_Type == 2:
        demod_Seq = ((receiveSeq < 0) * 1)
    if mod_Type == 4:
        demod_Seq = np.zeros((num_Symbol * 2, 1))
        for i in range(num_Symbol):
            demod_Seq[i * 2] = (receiveSeq[i].real > 0) * 1
            demod_Seq[i * 2 + 1] = (receiveSeq[i].imag < 0) * 1
            i = i + 1
    return demod_Seq


# source = np.random.randint(0, 2, (10, 1))
# a=modulation(source,5,modType,modName)
# a_demod=demodulation(a,5,modType)
# print(a)
# print(source)
# print(a_demod)
#
# print("end")









# Define the model of the system
x = tf.placeholder(tf.float32, shape=(None, 2 * K), name="transmit")
y = tf.placeholder(tf.float32, shape=(None, 2 * N), name="receiver")
realH = tf.placeholder(tf.float32, shape=(2*N,2*K), name="channel")

# Define the parameters of the network
# layer1
with tf.name_scope('layer1parameter'):
    w10 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b10 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    # b10=tf.Variable(tf.zeros([5*K,1]))
    # w20 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    # b20 = tf.Variable(tf.constant(0.1, shape=[2 * K, 1]))
    # w30=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
    # w30=tf.Variable(tf.zeros([2*K,5*K]))
    # b30=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
    tf.summary.histogram("weights1", w10)
    # tf.summary.histogram("weigths2", w20)
    # tf.summary.histogram("weights3",w30)
    tf.summary.histogram("biases1", b10)
    # tf.summary.histogram("biases2", b20)
# tf.summary.histogram("biases3",b30)

# layer2
with tf.name_scope('layer2parameter'):
    # w31=tf.Variable(tf.zeros([2*K,5*K]))
    w11 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b11 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    # w21 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    # b21 = tf.Variable(tf.constant(0.1, shape=[2 * K, 1]))
    # w31=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
    # b31=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
    tf.summary.histogram("weights1", w11)
    # tf.summary.histogram("weigths2", w21)
    # tf.summary.histogram("weights3",w31)
    tf.summary.histogram("biases1", b11)
    # tf.summary.histogram("biases2", b21)
# tf.summary.histogram("biases3",b31)

# layer3
with tf.name_scope('layer3parameter'):
    # w32=tf.Variable(tf.zeros([2*K,5*K]))
    w12 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b12 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    # w22 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    # b22 = tf.Variable(tf.constant(0.1, shape=[2 * K, 1]))
    # w32=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
    # b32=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
    tf.summary.histogram("weights1", w12)
    # tf.summary.histogram("weigths2", w22)
    # tf.summary.histogram("weights3", w32)
    tf.summary.histogram("biases1", b12)
    # tf.summary.histogram("biases2", b22)
# tf.summary.histogram("biases3", b32)
#
# layer4
with tf.name_scope('layer4parameter'):
    # w33=tf.Variable(tf.zeros([2*K,5*K]))
    w13 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b13 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    # w23 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    # b23 = tf.Variable(tf.constant(0.1, shape=[2 * K, 1]))
    # w33=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
    # b33=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
    tf.summary.histogram("weights1", w13)
    # tf.summary.histogram("weigths2", w23)
    # tf.summary.histogram("weights3", w33)
    tf.summary.histogram("biases1", b13)
    # tf.summary.histogram("biases2", b23)
# tf.summary.histogram("biases3", b33)

# layer5
with tf.name_scope('layer5parameter'):
    # w33=tf.Variable(tf.zeros([2*K,5*K]))
    w14 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b14 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    # w24 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    # b24 = tf.Variable(tf.constant(0.1, shape=[2 * K, 1]))
    # w33=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
    # b33=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
    tf.summary.histogram("weights1", w14)
    # tf.summary.histogram("weigths2", w24)
    # tf.summary.histogram("weights3", w33)
    tf.summary.histogram("biases1", b14)
    # tf.summary.histogram("biases2", b24)
# tf.summary.histogram("biases3", b33)

# layer6
with tf.name_scope('layer6parameter'):
    # w33=tf.Variable(tf.zeros([2*K,5*K]))
    w15 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b15 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    # w25 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    # b25 = tf.Variable(tf.constant(0.1, shape=[2 * K, 1]))
    # w33=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
    # b33=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
    tf.summary.histogram("weights1", w15)
    # tf.summary.histogram("weigths2", w25)
    # tf.summary.histogram("weights3", w33)
    tf.summary.histogram("biases1", b15)
    # tf.summary.histogram("biases2", b25)
# tf.summary.histogram("biases3", b33)

# layer7
with tf.name_scope('layer7parameter'):
    w16 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b16 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    tf.summary.histogram("weights1", w16)
    tf.summary.histogram("biases1", b16)

# layer8
with tf.name_scope('layer8parameter'):
    w17 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b17 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    tf.summary.histogram("weights1", w17)
    tf.summary.histogram("biases1", b17)

# layer9
with tf.name_scope('layer9parameter'):
    w18 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b18 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    tf.summary.histogram("weights1", w18)
    tf.summary.histogram("biases1", b18)

# layer10
with tf.name_scope('layer10parameter'):
    w19 = tf.Variable(tf.truncated_normal([2 * K, 6 * K], stddev=1, seed=1))
    b19 = tf.Variable(tf.constant(0.001, shape=[2 * K, 1]))
    tf.summary.histogram("weights1", w19)
    tf.summary.histogram("biases1", b19)



# Define the process of the forward propagation
# layer1
with tf.name_scope('layer1'):
    combination1 = tf.matmul(tf.transpose(realH), tf.transpose(y))
    s = tf.shape(combination1)
    # v0=tf.zeros([2*K,s[1]])
    x0 = tf.zeros([2 * K, s[1]])
    HtrH = tf.matmul(tf.transpose(realH), realH)
    combination2 = tf.matmul(HtrH, x0)
    concatenation = tf.concat([combination1, x0, combination2], 0, name="Concatenate")
    cal1 = tf.matmul(w10, concatenation) + b10+x0
    # z0 = tf.nn.relu(cal1, name="Z0")
    # z0=cal1
    # cal2 = tf.matmul(w20, z0) + b20 + x0
    t0 = tf.Variable(0.5, name='t0')
    x1 = -1 + tf.nn.relu(cal1 + t0) / abs(t0) - tf.nn.relu(cal1 - t0) / abs(t0)
# v1=tf.matmul(w30,z0)+b30

# layer2
with tf.name_scope('layer2'):
    combination2_1 = tf.matmul(HtrH, x1)
    concatenation_1 = tf.concat([combination1, x1, combination2_1], 0, name="Concatenate1")
    cal1_1 = tf.matmul(w11, concatenation_1) + b11+x1
    # z1 = tf.nn.relu(cal1_1, name="Z1")
    # z1=cal1_1
    # cal2_1 = tf.matmul(w21, z1) + b21 + x1
    t1 = tf.Variable(0.5, name='t1')
    x2 = -1 + tf.nn.relu(cal1_1 + t1) / abs(t1) - tf.nn.relu(cal1_1 - t1) / abs(t1)
# v2=tf.matmul(w31,z1)+b31

# layer3
with tf.name_scope('layer3'):
    combination2_2 = tf.matmul(HtrH, x2)
    concatenation_2 = tf.concat([combination1, x2, combination2_2], 0, name="Concatenate2")
    cal1_2 = tf.matmul(w12, concatenation_2) + b12+x2
    # z2 = tf.nn.relu(cal1_2, name="Z2")
    # z2=cal1_2
    # cal2_2 = tf.matmul(w22, z2) + b22 + x2
    t2 = tf.Variable(0.5, name='t2')
    x3 = -1 + tf.nn.relu(cal1_2 + t2) / abs(t2) - tf.nn.relu(cal1_2 - t2) / abs(t2)
# v3=tf.matmul(w32,z2)+b32
#
# layer4
with tf.name_scope('layer4'):
    combination2_3 = tf.matmul(HtrH, x3)
    concatenation_3 = tf.concat([combination1, x3, combination2_3], 0, name="Concatenate3")
    cal1_3 = tf.matmul(w13, concatenation_3) + b13+x3
    # z3 = tf.nn.relu(cal1_3, name="Z3")
    # z3=cal1_3
    # cal2_3 = tf.matmul(w23, z3) + b23 + x3
    t3 = tf.Variable(0.5, name='t3')
    x4 = -1 + tf.nn.relu(cal1_3 + t3) / abs(t3) - tf.nn.relu(cal1_3 - t3) / abs(t3)
# v4=tf.matmul(w33,z3)+b33

# layer5
with tf.name_scope('layer5'):
    combination2_4 = tf.matmul(HtrH, x4)
    concatenation_4 = tf.concat([combination1, x4, combination2_4], 0, name="Concatenate4")
    cal1_4 = tf.matmul(w14, concatenation_4) + b14+x4
    # z4 = tf.nn.relu(cal1_4, name="Z4")
    # z4=cal1_4
    # cal2_4 = tf.matmul(w24, z4) + b24 + x4
    t4 = tf.Variable(0.5, name='t4')
    x5 = -1 + tf.nn.relu(cal1_4 + t4) / abs(t4) - tf.nn.relu(cal1_4 - t4) / abs(t4)
# v4=tf.matmul(w33,z3)+b33

# layer6
with tf.name_scope('layer6'):
    combination2_5 = tf.matmul(HtrH, x5)
    concatenation_5 = tf.concat([combination1, x5, combination2_5], 0, name="Concatenate5")
    cal1_5 = tf.matmul(w15, concatenation_5) + b15+x5
    # z5 = tf.nn.relu(cal1_5, name="Z5")
    # z5=cal1_5
    # cal2_5 = tf.matmul(w25, z5) + b25 + x5
    t5 = tf.Variable(0.5, name='t5')
    x6 = -1 + tf.nn.relu(cal1_5 + t5) / abs(t5) - tf.nn.relu(cal1_5 - t5) / abs(t5)
# v4=tf.matmul(w33,z3)+b33

# layer7
with tf.name_scope('layer7'):
    combination2_6 = tf.matmul(HtrH, x6)
    concatenation_6 = tf.concat([combination1, x6, combination2_6], 0, name="Concatenate6")
    cal1_6 = tf.matmul(w16, concatenation_6) + b16+x6
    t6 = tf.Variable(0.5, name='t6')
    x7 = -1 + tf.nn.relu(cal1_6 + t6) / abs(t6) - tf.nn.relu(cal1_6 - t6) / abs(t6)

# layer8
with tf.name_scope('layer8'):
    combination2_7 = tf.matmul(HtrH, x7)
    concatenation_7 = tf.concat([combination1, x7, combination2_7], 0, name="Concatenate7")
    cal1_7 = tf.matmul(w17, concatenation_7) + b17+x7
    t7 = tf.Variable(0.5, name='t7')
    x8 = -1 + tf.nn.relu(cal1_7 + t7) / abs(t7) - tf.nn.relu(cal1_7 - t7) / abs(t7)

# layer9
with tf.name_scope('layer9'):
    combination2_8 = tf.matmul(HtrH, x8)
    concatenation_8 = tf.concat([combination1, x8, combination2_8], 0, name="Concatenate8")
    cal1_8 = tf.matmul(w18, concatenation_8) + b18+x8
    t8 = tf.Variable(0.5, name='t8')
    x9 = -1 + tf.nn.relu(cal1_8 + t8) / abs(t8) - tf.nn.relu(cal1_8 - t8) / abs(t8)

# layer10
with tf.name_scope('layer10'):
    combination2_9 = tf.matmul(HtrH, x9)
    concatenation_9 = tf.concat([combination1, x9, combination2_9], 0, name="Concatenate9")
    cal1_9 = tf.matmul(w19, concatenation_9) + b19+x9
    t9 = tf.Variable(0.5, name='t9')
    x10 = -1 + tf.nn.relu(cal1_9 + t9) / abs(t9) - tf.nn.relu(cal1_9 - t9) / abs(t9)



# Define loss function and backpropagation algorithm
with tf.name_scope('decorrelator'):
    xwave_part1 = tf.matrix_inverse(tf.matmul(tf.transpose(realH), realH))
    xwave_part2 = tf.matmul(xwave_part1, tf.transpose(realH))
    xwave = tf.matmul(xwave_part2, tf.transpose(y))

with tf.name_scope('loss'):
    lossfunction = tf.log(1.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x1)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(2.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x2)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(3.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x3)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(4.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x4)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(5.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x5)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(6.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x6)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(7.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x7)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(8.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x8)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(9.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x9)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave)) \
                   + tf.log(10.00) * tf.reduce_sum(tf.squared_difference(tf.transpose(x), x10)) / tf.reduce_sum(
        tf.squared_difference(tf.transpose(x), xwave))
    tf.summary.scalar("loss", lossfunction)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(lossfunction)

summ = tf.summary.merge_all()


# define function to generate data
def generate_data(batchsize, K, N, mod_Type, mod_Name):
    while 1:
        H = (np.random.randn(N, K) + 1j * np.random.randn(N, K)) / np.sqrt(2)
        U,S,V=la.svd(H)
        cond=max(S)/min(S)
        if cond > 7:
            break
    H = np.sqrt(N) / np.sqrt(np.trace(np.dot(np.transpose(np.conjugate(H)), H))) * H
    Hpart1 = np.concatenate((np.real(H), np.imag(H)))
    Hpart2 = np.concatenate((-np.imag(H), np.real(H)))
    H_real = np.concatenate((Hpart1, Hpart2), axis=1)
    H_real = H_real.astype('float32')
    symbolbits = np.log2(mod_Type)
    symbolbits = symbolbits.astype(int)
    source = np.random.randint(0, 2, ((batchsize * K) * symbolbits, 1))  # generate 0,1 bits
    x_complex = modulation(source, batchsize * K, mod_Type, mod_Name)  # QPSK modulation
    x_complex = np.reshape(x_complex, (batchsize, K))
    w = np.zeros((N, batchsize),
                 dtype='complex')  # Noise Vector with independent,zero mean Gaussian variables of variance 1
    for m in range(batchsize):
        SNR = np.random.uniform(8, 16, 1)  ##8dB-13dB uniform distribution
        sigma = np.sqrt(1 / (10 ** (SNR / 10)))
        wpart = sigma / np.sqrt(2) * np.random.randn(N) + 1j * sigma / np.sqrt(2) * np.random.randn(N)
        w[:, m] = wpart
    # x_ = x_.astype('float32')
    y_ = np.dot(H, np.transpose(x_complex)) + w
    y_complex = np.transpose(y_)
    x_real = np.concatenate((np.real(x_complex), np.imag(x_complex)), axis=1)
    y_real = np.concatenate((np.real(y_complex), np.imag(y_complex)), axis=1)
    return source, x_real, y_real, H_real


def generate_testdata(symbolnum, K, N, mod_Type, mod_Name, SNR):
    while 1:
        H = (np.random.randn(N, K) + 1j * np.random.randn(N, K)) / np.sqrt(2)
        U,S,V=la.svd(H)
        cond=max(S)/min(S)
        if cond > 7:
            break
    # H = (np.random.randn(N,K)+1j*np.random.randn(N,K))/np.sqrt(2)
    H = np.sqrt(N) / np.sqrt(np.trace(np.dot(np.transpose(np.conjugate(H)), H))) * H
    Hpart1 = np.concatenate((np.real(H), np.imag(H)))
    Hpart2 = np.concatenate((-np.imag(H), np.real(H)))
    H_real = np.concatenate((Hpart1, Hpart2), axis=1)
    H_real = H_real.astype('float32')
    symbolbits = np.log2(mod_Type)
    symbolbits = symbolbits.astype(int)
    # np.random.seed(1)
    source = np.random.randint(0, 2, ((symbolnum * K) * symbolbits, 1))  # generate 0,1 bits
    x_complex = modulation(source, symbolnum * K, mod_Type, mod_Name)
    x_complex = np.reshape(x_complex, (symbolnum, K))
    w = np.zeros((N, symbolnum), dtype='complex')
    sigma = np.sqrt(1 / (10 ** (SNR / 10)))
    for m in range(symbolnum):
        wpart = sigma / np.sqrt(2) * np.random.randn(N) + 1j * sigma / np.sqrt(2) * np.random.randn(N)
        w[:, m] = wpart
    # x_ = x_.astype('float32')
    y_ = np.dot(H, np.transpose(x_complex)) + w
    y_complex = np.transpose(y_)
    x_real = np.concatenate((np.real(x_complex), np.imag(x_complex)), axis=1)
    y_real = np.concatenate((np.real(y_complex), np.imag(y_complex)), axis=1)
    return source,  x_real, y_real, H_real


# Create a session to run Tensorflow
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

for i in range(STEPS):
    source, x_, y_ ,H_= generate_data(batch_size, K, N, modType, modName)
    sess.run(train_step,
             feed_dict={x: x_, y: y_,realH: H_})  # train wk bk

    if i % 500 == 0:
        s = sess.run(summ, feed_dict={x: x_, y: y_,realH:H_})
        writer.add_summary(s, i)
    if i % 5000 == 0:
        total_lossfunction = sess.run(lossfunction, feed_dict={x: x_, y: y_,realH:H_})
        print("After %d training steps,cross entropy on all data is %g" % (i, total_lossfunction))
        print("x10:", sess.run(x10, feed_dict={y: y_,realH:H_}))
        # print("v3:",sess.run(v3,feed_dict={y:y_}))
        # print("z3:",sess.run(z3,feed_dict={y:y_}))
        print("b11:", sess.run(b11))
        # print("b21:", sess.run(b21))
        # print("b31:", sess.run(b31))
        # print("b32:", sess.run(b32))
        # print("w31:", sess.run(w31))
        # print("w32:", sess.run(w32))
        print("t0:", sess.run(t0))
        print("t1:", sess.run(t1))
        print("t2:", sess.run(t2))
        print("t3:", sess.run(t3))

print("w10:", sess.run(w10))
print("w11:", sess.run(w11))
print("w12:", sess.run(w12))
print("w13:", sess.run(w13))

testsymbolnum = 1000
Hnum=10000
symbolbits = np.log2(modType)
symbolbits = symbolbits.astype(int)
sourcelength = testsymbolnum * symbolbits * K*Hnum
xkout2 = np.zeros((2 * K, testsymbolnum))
xkout4 = np.zeros((2 * K, testsymbolnum))
xkout6 = np.zeros((2 * K, testsymbolnum))
xkout8 = np.zeros((2 * K, testsymbolnum))
xkout10 = np.zeros((2 * K, testsymbolnum))
layernum = 5
ber = np.zeros((layernum, 8))
berdecorSet = np.zeros((1, 8))
indexcount = 0

for i1 in range(8, 16):
    testbiterrorsum2=0
    testbiterrorsum4=0
    testbiterrorsum6=0
    testbiterrorsum8=0
    testbiterrorsum10=0
    biterrorsum_decor_test=0
    for j1 in range (Hnum):
        source_test,  x_test, y_test, H_test = generate_testdata(testsymbolnum, K, N,modType,modName, i1)
        xkout2 = sess.run(x2, feed_dict={y: y_test,realH:H_test})
        xkout4 = sess.run(x4, feed_dict={y: y_test,realH:H_test})
        xkout6 = sess.run(x6, feed_dict={y: y_test,realH:H_test})
        xkout8 = sess.run(x8, feed_dict={y: y_test,realH:H_test})
        xkout10 = sess.run(x10, feed_dict={y: y_test,realH:H_test})

        # xkout1_complex = xkout1[0:K, :] + 1j * xkout1[K:2 * K, :]
        # xkout1_complex_seq = np.reshape(np.transpose(xkout1_complex), (testsymbolnum * K, 1))
        xkout2_complex = xkout2[0:K, :] + 1j * xkout2[K:2 * K, :]
        xkout2_complex_seq = np.reshape(np.transpose(xkout2_complex), (testsymbolnum * K, 1))
        # xkout3_complex = xkout3[0:K, :] + 1j * xkout3[K:2 * K, :]
        # xkout3_complex_seq = np.reshape(np.transpose(xkout3_complex), (testsymbolnum * K, 1))
        xkout4_complex = xkout4[0:K, :] + 1j * xkout4[K:2 * K, :]
        xkout4_complex_seq = np.reshape(np.transpose(xkout4_complex), (testsymbolnum * K, 1))
        # xkout5_complex = xkout5[0:K, :] + 1j * xkout5[K:2 * K, :]
        # xkout5_complex_seq = np.reshape(np.transpose(xkout5_complex), (testsymbolnum * K, 1))
        xkout6_complex = xkout6[0:K, :] + 1j * xkout6[K:2 * K, :]
        xkout6_complex_seq = np.reshape(np.transpose(xkout6_complex), (testsymbolnum * K, 1))
        xkout8_complex = xkout8[0:K, :] + 1j * xkout8[K:2 * K, :]
        xkout8_complex_seq = np.reshape(np.transpose(xkout8_complex), (testsymbolnum * K, 1))
        xkout10_complex = xkout10[0:K, :] + 1j * xkout10[K:2 * K, :]
        xkout10_complex_seq = np.reshape(np.transpose(xkout10_complex), (testsymbolnum * K, 1))

        xkout2bit_seq = demodulation(xkout2_complex_seq, testsymbolnum * K, modType)
        xkout4bit_seq = demodulation(xkout4_complex_seq, testsymbolnum * K, modType)
        xkout6bit_seq = demodulation(xkout6_complex_seq, testsymbolnum * K, modType)
        xkout8bit_seq = demodulation(xkout8_complex_seq, testsymbolnum * K, modType)
        xkout10bit_seq = demodulation(xkout10_complex_seq, testsymbolnum * K, modType)

        testbiterrorsum2 = testbiterrorsum2 + np.sum(abs(xkout2bit_seq - source_test))
        testbiterrorsum4 = testbiterrorsum4 + np.sum(abs(xkout4bit_seq - source_test))
        testbiterrorsum6 = testbiterrorsum6 + np.sum(abs(xkout6bit_seq - source_test))
        testbiterrorsum8 = testbiterrorsum8 + np.sum(abs(xkout8bit_seq - source_test))
        testbiterrorsum10 = testbiterrorsum10 + np.sum(abs(xkout10bit_seq - source_test))


        xdecor = sess.run(xwave, feed_dict={y: y_test, realH: H_test})
        xdecor_complex = xdecor[0:K, :] + 1j * xdecor[K:2 * K, :]
        xdecor_complex_seq = np.reshape(np.transpose(xdecor_complex), (testsymbolnum * K, 1))
        xdecor_seq = demodulation(xdecor_complex_seq, testsymbolnum * K, modType)
        biterrorsum_decor_test = biterrorsum_decor_test + np.sum(abs(xdecor_seq - source_test))


    ber[:, indexcount] = [testbiterrorsum2/sourcelength, testbiterrorsum4/sourcelength, testbiterrorsum6/sourcelength,testbiterrorsum8/sourcelength,testbiterrorsum10/sourcelength]
    berdecorSet[:, indexcount] = biterrorsum_decor_test / sourcelength
    indexcount = indexcount + 1

print("ber:", ber)
print("ber_decorSet", berdecorSet)
scipy.io.savemat('result/layernum10/16x8/simplify/limit7/ber.mat', {'ber': ber})
scipy.io.savemat('result/layernum10/16x8/simplify/limit7/xkout10_complex_seq.mat', {'xkout10_complex_seq': xkout10_complex_seq})
scipy.io.savemat('result/layernum10/16x8/simplify/limit7/x10.mat', {'x10': xkout10bit_seq})
scipy.io.savemat('result/layernum10/16x8/simplify/limit7/source.mat', {'source': source_test})
scipy.io.savemat('result/layernum10/16x8/simplify/limit7/berdecorSet.mat', {'berdecorSet': berdecorSet})

sess.close()
print("layernum10_simplify_5000_16*8_end")
print("end")
