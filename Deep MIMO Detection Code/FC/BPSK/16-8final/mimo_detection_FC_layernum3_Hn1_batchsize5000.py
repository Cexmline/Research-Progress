import tensorflow as tf
import numpy as np
import scipy.io


LOGDIR="path/layernum3/Hn1/batchsize5000-steps50000"
K=8  #Number of users
N=16 #Number of receiving antenna
batch_size=5000 #Define the batch size
STEPS=50000  #Number of iteration
Hdata=scipy.io.loadmat('Hn1_10.1821')
H=Hdata['Hn1']
H=H.astype('float32')


#Define the model of the system
x=tf.placeholder(tf.float32,shape=(None,K),name="transmit")
y=tf.placeholder(tf.float32,shape=(None,N),name="receiver")

#Define the parameters of the network
#layer1
with tf.name_scope('layer1parameter'):
	w10=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
	b10=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
	# b10=tf.Variable(tf.zeros([5*K,1]))
	w20=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
	b20=tf.Variable(tf.constant(0.1,shape=[K,1]))
	w30=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	#w30=tf.Variable(tf.zeros([2*K,5*K]))
	b30=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1",w10)
	tf.summary.histogram("weigths2",w20)
	tf.summary.histogram("weights3",w30)
	tf.summary.histogram("biases1",b10)
	tf.summary.histogram("biases2",b20)
	tf.summary.histogram("biases3",b30)

#layer2
with tf.name_scope('layer2parameter'):
	#w31=tf.Variable(tf.zeros([2*K,5*K]))
	w11=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
	b11=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
	w21=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
	b21=tf.Variable(tf.constant(0.1,shape=[K,1]))
	w31=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	b31=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1",w11)
	tf.summary.histogram("weigths2",w21)
	tf.summary.histogram("weights3",w31)
	tf.summary.histogram("biases1",b11)
	tf.summary.histogram("biases2",b21)
	tf.summary.histogram("biases3",b31)

#layer3
with tf.name_scope('layer3parameter'):
	#w32=tf.Variable(tf.zeros([2*K,5*K]))
	w12=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
	b12=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
	w22=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
	b22=tf.Variable(tf.constant(0.1,shape=[K,1]))
	w32=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	b32=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1", w12)
	tf.summary.histogram("weigths2", w22)
	tf.summary.histogram("weights3", w32)
	tf.summary.histogram("biases1", b12)
	tf.summary.histogram("biases2", b22)
	tf.summary.histogram("biases3", b32)

# #layer4
# with tf.name_scope('layer4parameter'):
# 	#w33=tf.Variable(tf.zeros([2*K,5*K]))
# 	w13=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
# 	b13=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
# 	w23=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
# 	b23=tf.Variable(tf.constant(0.1,shape=[K,1]))
# 	w33=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
# 	b33=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
# 	tf.summary.histogram("weights1", w13)
# 	tf.summary.histogram("weigths2", w23)
# 	tf.summary.histogram("weights3", w33)
# 	tf.summary.histogram("biases1", b13)
# 	tf.summary.histogram("biases2", b23)
# 	tf.summary.histogram("biases3", b33)

# #layer5
# with tf.name_scope('layer5parameter'):
# 	# w34=tf.Variable(tf.zeros([2*K,5*K]))
# 	w14=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
# 	b14=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
# 	w24=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
# 	b24=tf.Variable(tf.constant(0.1,shape=[K,1]))
# 	w34=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
# 	b34=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
# 	tf.summary.histogram("weights1", w14)
# 	tf.summary.histogram("weigths2", w24)
# 	tf.summary.histogram("weights3", w34)
# 	tf.summary.histogram("biases1", b14)
# 	tf.summary.histogram("biases2", b24)
# 	tf.summary.histogram("biases3", b34)
#
# #layer6
# with tf.name_scope('layer6parameter'):
# 	# w35=tf.Variable(tf.zeros([2*K,5*K]))
# 	w15=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
# 	b15=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
# 	w25=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
# 	b25=tf.Variable(tf.constant(0.1,shape=[K,1]))
# 	w35=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
# 	b35=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
# 	tf.summary.histogram("weights1", w15)
# 	tf.summary.histogram("weigths2", w25)
# 	tf.summary.histogram("weights3", w35)
# 	tf.summary.histogram("biases1", b15)
# 	tf.summary.histogram("biases2", b25)
# 	tf.summary.histogram("biases3", b35)

# #layer7
# with tf.name_scope('layer7parameter'):
# 	#w36=tf.Variable(tf.zeros([2*K,5*K]))
# 	w16=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
# 	b16=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
# 	w26=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
# 	b26=tf.Variable(tf.constant(0.1,shape=[K,1]))
# 	w36=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
# 	b36=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
# 	tf.summary.histogram("weights1", w16)
# 	tf.summary.histogram("weigths2", w26)
# 	tf.summary.histogram("weights3", w36)
# 	tf.summary.histogram("biases1", b16)
# 	tf.summary.histogram("biases2", b26)
# 	tf.summary.histogram("biases3", b36)
# #
# #layer8
# with tf.name_scope('layer8parameter'):
# 	#w37=tf.Variable(tf.zeros([2*K,5*K]))
# 	w17=tf.Variable(tf.truncated_normal([5*K,5*K],stddev=1,seed=1))
# 	b17=tf.Variable(tf.constant(0.1,shape=[5*K,1]))
# 	w27=tf.Variable(tf.truncated_normal([K,5*K],stddev=1,seed=1))
# 	b27=tf.Variable(tf.constant(0.1,shape=[K,1]))
# 	w37=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
# 	b37=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
#
# #layer9
# w38=tf.Variable(tf.zeros([2*K,5*K]))
# w18=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
# b18=tf.Variable(tf.zeros([5*K,1]))
# w28=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
# b28=tf.Variable(tf.zeros([K,1]))
# #w38=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
# b38=tf.Variable(tf.zeros([2*K,1]))
#
# #layer10
# w39=tf.Variable(tf.zeros([2*K,5*K]))
# w19=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
# b19=tf.Variable(tf.zeros([5*K,1]))
# w29=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
# b29=tf.Variable(tf.zeros([K,1]))
# #w39=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
# b39=tf.Variable(tf.zeros([2*K,1]))
#
# #layer11
# w310=tf.Variable(tf.zeros([2*K,5*K]))
# w110=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
# b110=tf.Variable(tf.zeros([5*K,1]))
# w210=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
# b210=tf.Variable(tf.zeros([K,1]))
# #w310=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
# b310=tf.Variable(tf.zeros([2*K,1]))
#
# #layer12
# w311=tf.Variable(tf.zeros([2*K,5*K]))
# w111=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
# b111=tf.Variable(tf.zeros([5*K,1]))
# w211=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
# b211=tf.Variable(tf.zeros([K,1]))
# #w311=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
# b311=tf.Variable(tf.zeros([2*K,1]))
#




#Define the process of the forward propagation 
#layer1
with tf.name_scope('layer1'):
	combination1=tf.matmul(tf.transpose(H),tf.transpose(y))
	s=tf.shape(combination1)
	v0=tf.zeros([2*K,s[1]])
	x0=tf.zeros([K,s[1]])
	HtrH=tf.matmul(tf.transpose(H),H)
	combination2=tf.matmul(HtrH,x0)
	concatenation=tf.concat([combination1,x0,combination2,v0],0,name="Concatenate")
	cal1=tf.matmul(w10,concatenation)+b10
	z0=tf.nn.relu(cal1,name="Z0")
	cal2=tf.matmul(w20,z0)+b20+x0
	t0=tf.Variable(0.5,name='t0')
	x1=-1+tf.nn.relu(cal2+t0)/abs(t0)-tf.nn.relu(cal2-t0)/abs(t0)
	v1=tf.matmul(w30,z0)+b30

#layer2
with tf.name_scope('layer2'):
	combination2_1=tf.matmul(HtrH,x1)
	concatenation_1=tf.concat([combination1,x1,combination2_1,v1],0,name="Concatenate1")
	cal1_1=tf.matmul(w11,concatenation_1)+b11
	z1=tf.nn.relu(cal1_1,name="Z1")
	# k1 = tf.Variable(0.5, name='k1')
	cal2_1=tf.matmul(w21,z1)+b21+x1
	t1=tf.Variable(0.5,name='t1')
	x2=-1+tf.nn.relu(cal2_1+t1)/abs(t1)-tf.nn.relu(cal2_1-t1)/abs(t1)
	v2=tf.matmul(w31,z1)+b31

#layer3
with tf.name_scope('layer3'):
	combination2_2=tf.matmul(HtrH,x2)
	concatenation_2=tf.concat([combination1,x2,combination2_2,v2],0,name="Concatenate2")
	cal1_2=tf.matmul(w12,concatenation_2)+b12
	z2=tf.nn.relu(cal1_2,name="Z2")
	# k2 = tf.Variable(0.5, name='k2')
	cal2_2=tf.matmul(w22,z2)+b22+x2
	t2=tf.Variable(0.5,name='t2')
	x3=-1+tf.nn.relu(cal2_2+t2)/abs(t2)-tf.nn.relu(cal2_2-t2)/abs(t2)
	v3=tf.matmul(w32,z2)+b32

# #layer4
# with tf.name_scope('layer4'):
# 	combination2_3=tf.matmul(HtrH,x3)
# 	concatenation_3=tf.concat([combination1,x3,combination2_3,v3],0,name="Concatenate3")
# 	cal1_3=tf.matmul(w13,concatenation_3)+b13
# 	z3=tf.nn.relu(cal1_3,name="Z3")
# 	# k3 = tf.Variable(0.5, name='k3')
# 	cal2_3=tf.matmul(w23,z3)+b23+x3
# 	t3=tf.Variable(0.5,name='t3')
# 	x4=-1+tf.nn.relu(cal2_3+t3)/abs(t3)-tf.nn.relu(cal2_3-t3)/abs(t3)
# 	v4=tf.matmul(w33,z3)+b33

# #layer5
# with tf.name_scope('layer5'):
# 	combination2_4=tf.matmul(HtrH,x4)
# 	concatenation_4=tf.concat([combination1,x4,combination2_4,v4],0,name="Concatenate4")
# 	cal1_4=tf.matmul(w14,concatenation_4)+b14
# 	z4=tf.nn.relu(cal1_4,name="Z4")
# 	# k4 = tf.Variable(0.5, name='k4')
# 	cal2_4=tf.matmul(w24,z4)+b24+x4
# 	t4=tf.Variable(0.5,name='t4')
# 	x5=-1+tf.nn.relu(cal2_4+t4)/abs(t4)-tf.nn.relu(cal2_4-t4)/abs(t4)
# 	v5=tf.matmul(w34,z4)+b34
#
# #layer6
# with tf.name_scope('layer6'):
# 	combination2_5=tf.matmul(HtrH,x5)
# 	concatenation_5=tf.concat([combination1,x5,combination2_5,v5],0,name="Concatenate5")
# 	cal1_5=tf.matmul(w15,concatenation_5)+b15
# 	z5=tf.nn.relu(cal1_5,name="Z5")
# 	# k5 = tf.Variable(0.5, name='k5')
# 	cal2_5=tf.matmul(w25,z5)+b25+x5
# 	t5=tf.Variable(0.5,name='t5')
# 	x6=-1+tf.nn.relu(cal2_5+t5)/abs(t5)-tf.nn.relu(cal2_5-t5)/abs(t5)
# 	v6=tf.matmul(w35,z5)+b35

# #layer7
# with tf.name_scope('layer7'):
# 	combination2_6=tf.matmul(HtrH,x6)
# 	concatenation_6=tf.concat([combination1,x6,combination2_6,v6],0,name="Concatenate6")
# 	cal1_6=tf.matmul(w16,concatenation_6)+b16
# 	z6=tf.nn.relu(cal1_6,name="Z6")
# 	cal2_6=tf.matmul(w26,z6)+b26+x6
# 	t6=tf.Variable(0.5,name='t6')
# 	x7=-1+tf.nn.relu(cal2_6+t6)/abs(t6)-tf.nn.relu(cal2_6-t6)/abs(t6)
# 	v7=tf.matmul(w36,z6)+b36

# #layer8
# with tf.name_scope('layer8'):
# 	combination2_7=tf.matmul(HtrH,x7)
# 	concatenation_7=tf.concat([combination1,x7,combination2_7,v7],0,name="Concatenate7")
# 	cal1_7=tf.matmul(w17,concatenation_7)+b17
# 	z7=tf.nn.relu(cal1_7,name="Z7")
# 	cal2_7=tf.matmul(w27,z7)+b27+x7
# 	t7=tf.Variable(0.5,name='t7')
# 	x8=-1+tf.nn.relu(cal2_7+t7)/abs(t7)-tf.nn.relu(cal2_7-t7)/abs(t7)
# 	v8=tf.matmul(w37,z7)+b37

# #layer9
# combination2_8=tf.matmul(HtrH,x8)
# concatenation_8=tf.concat([combination1,x8,combination2_8,v8],0,name="Concatenate8")
# cal1_8=tf.matmul(w18,concatenation_8)+b18
# z8=tf.nn.relu(cal1_8,name="Z8")
# cal2_8=tf.matmul(w28,z8)+b28+x8
# t8=tf.Variable(0.5,name='t8')
# x9=-1+tf.nn.relu(cal2_8+t8)/abs(t8)-tf.nn.relu(cal2_8-t8)/abs(t8)
# v9=tf.matmul(w38,z8)+b38
#
# #layer10
# combination2_9=tf.matmul(HtrH,x9)
# concatenation_9=tf.concat([combination1,x9,combination2_9,v9],0,name="Concatenate9")
# cal1_9=tf.matmul(w19,concatenation_9)+b19
# z9=tf.nn.relu(cal1_9,name="Z9")
# cal2_9=tf.matmul(w29,z9)+b29+x9
# t9=tf.Variable(0.5,name='t9')
# x10=-1+tf.nn.relu(cal2_9+t9)/abs(t9)-tf.nn.relu(cal2_9-t9)/abs(t9)
# v10=tf.matmul(w39,z9)+b39
#
# #layer11
# combination2_10=tf.matmul(HtrH,x10)
# concatenation_10=tf.concat([combination1,x10,combination2_10,v10],0,name="Concatenate10")
# cal1_10=tf.matmul(w110,concatenation_10)+b110
# z10=tf.nn.relu(cal1_10,name="Z10")
# cal2_10=tf.matmul(w210,z10)+b210+x10
# t10=tf.Variable(0.5,name='t10')
# x11=-1+tf.nn.relu(cal2_10+t10)/abs(t10)-tf.nn.relu(cal2_10-t10)/abs(t10)
# v11=tf.matmul(w310,z10)+b310
#
#
# #layer12
# combination2_11=tf.matmul(HtrH,x11)
# concatenation_11=tf.concat([combination1,x11,combination2_11,v11],0,name="Concatenate11")
# cal1_11=tf.matmul(w111,concatenation_11)+b111
# z11=tf.nn.relu(cal1_11,name="Z11")
# cal2_11=tf.matmul(w211,z11)+b211+x11
# t11=tf.Variable(0.5,name='t11')
# x12=-1+tf.nn.relu(cal2_11+t11)/abs(t11)-tf.nn.relu(cal2_11-t11)/abs(t11)
# v12=tf.matmul(w311,z11)+b311
#
#

#Define loss function and backpropagation algorithm
with tf.name_scope('decorrelator'):
	xwave_part1=tf.matrix_inverse(tf.matmul(tf.transpose(H),H))
	xwave_part2=tf.matmul(xwave_part1,tf.transpose(H))
	xwave=tf.matmul(xwave_part2,tf.transpose(y))

with tf.name_scope('loss'):
	lossfunction=tf.log(1.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x1))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
			 	+tf.log(2.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x2))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
             	+tf.log(3.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x3))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))
             	# +tf.log(4.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x4))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))
				# +tf.log(5.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x5))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
				# +tf.log(6.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x6))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))
				# +tf.log(7.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x7))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))
				# +tf.log(8.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x8))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))
				 # +tf.log(9.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x9))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
				 # +tf.log(10.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x10))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
				 # +tf.log(11.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x11))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
				 # +tf.log(12.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x12))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
	tf.summary.scalar("loss",lossfunction)

with tf.name_scope('train'):
	train_step=tf.train.AdamOptimizer(0.001).minimize(lossfunction)

summ=tf.summary.merge_all()


# define function to generate data
def generate_data(batchsize,K,N,H):
	source = np.random.randint(0, 2, (batchsize * K, 1))  # generate 0,1 bits
	x_ = -2.0 * source + 1.0       # BPSK modulation
	x_ = np.reshape(x_, (batchsize, K))
	w = np.zeros((N, batchsize))  # Noise Vector with independent,zero mean Gaussian variables of variance 1
	for j in range(batchsize):
		SNR = np.random.uniform(8, 13, 1)  ##8dB-13dB uniform distribution
		sigma = np.sqrt(1 / (10 ** (SNR / 10)))
		wpart = sigma * np.random.randn(N)
		w[:, j] = wpart
	x_ = x_.astype('float32')
	y_ = np.dot(H, np.transpose(x_)) + w
	y_ = np.transpose(y_)
	return source, x_, y_

def generate_testdata(symbolnum,K,N,H,SNR):
	np.random.seed(1)
	source = np.random.randint(0, 2, (symbolnum * K, 1))  # generate 0,1 bits
	x_ = -2.0 * source + 1.0       # BPSK modulation
	x_ = np.reshape(x_, (symbolnum, K))
	w = np.zeros((N, symbolnum))  # Noise Vector with independent,zero mean Gaussian variables of variance 1
	sigma = np.sqrt(1 / (10 ** (SNR / 10)))
	for j in range(symbolnum):
		wpart = sigma * np.random.randn(N)
		w[:, j] = wpart
	x_ = x_.astype('float32')
	y_ = np.dot(H, np.transpose(x_)) + w
	y_ = np.transpose(y_)
	return source, x_, y_



#Create a session to run Tensorflow
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)
writer=tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

for i in range(STEPS):
	source, x_, y_ = generate_data(batch_size,K,N,H)
	sess.run(train_step,
			feed_dict={x:x_,y:y_}) #train wk bk


	if i%500==0:
		s=sess.run(summ,feed_dict={x:x_,y:y_})
		writer.add_summary(s,i)
	if i%5000==0:
		total_lossfunction=sess.run(lossfunction,feed_dict={x:x_,y:y_})
		print("After %d training steps,cross entropy on all data is %g"%(i,total_lossfunction))
		print("x3:",sess.run(x3,feed_dict={y:y_}))
		print("v3:",sess.run(v3,feed_dict={y:y_}))
		# print("z3:",sess.run(z3,feed_dict={y:y_}))
		print("b11:", sess.run(b11))
		print("b21:", sess.run(b21))
		print("b31:", sess.run(b31))
		print("b32:", sess.run(b32))
		print("w31:", sess.run(w31))
		print("w32:", sess.run(w32))
		print("t0:",sess.run(t0))
		print("t1:", sess.run(t1))
		print("t2:", sess.run(t2))
		# print("t3:", sess.run(t3))
		# print("t4:", sess.run(t4))
		# print("t5:", sess.run(t5))
		# print("k1:", sess.run(k1))
		# print("k2:", sess.run(k2))
		# print("k3:", sess.run(k3))
		# print("k4:", sess.run(k4))
		# print("k5:", sess.run(k5))
		# print("t4:", sess.run(t4))
		# print("t5:", sess.run(t5))
		# print("t6:", sess.run(t6))
		# print("t7:", sess.run(t7))
		# print("t8:", sess.run(t8))
		# print("t9:", sess.run(t9))
		# print("t10:", sess.run(t10))
		# print("t11:", sess.run(t11))
		# print("t12:", sess.run(t12))
		# print("t13:", sess.run(t13))
		# print("t14:", sess.run(t14))
		# print("t15:", sess.run(t15))
		# print("t16:", sess.run(t16))
		# print("t17:", sess.run(t17))
		# print("t18:", sess.run(t18))
		# print("t19:", sess.run(t19))







testsymbolnum=1000000
xkout1=np.zeros((K,testsymbolnum))
xkout2=np.zeros((K,testsymbolnum))
xkout3=np.zeros((K,testsymbolnum))
# xkout4=np.zeros((K,testsymbolnum))
# xkout5=np.zeros((K,testsymbolnum))
# xkout6=np.zeros((K,testsymbolnum))
# xkout7=np.zeros((K,testsymbolnum))
# xkout8=np.zeros((K,testsymbolnum))
# xkout9=np.zeros((K,testsymbolnum))
# xkout10=np.zeros((K,testsymbolnum))
# xkout11=np.zeros((K,testsymbolnum))
# xkout12=np.zeros((K,testsymbolnum))
# xkout13=np.zeros((K,testsymbolnum))
# xkout14=np.zeros((K,testsymbolnum))
# xkout15=np.zeros((K,testsymbolnum))
# xkout16=np.zeros((K,testsymbolnum))







for i1 in range (8,14):
	source_test,x_test,y_test=generate_testdata(testsymbolnum, K, N, H, i1)
	xkout1 = sess.run(x1, feed_dict={y: y_test})
	xkout2 = sess.run(x2, feed_dict={y: y_test})
	xkout3 = sess.run(x3, feed_dict={y: y_test})
	# xkout4 = sess.run(x4, feed_dict={y: y_test})
	# xkout5 = sess.run(x5, feed_dict={y: y_test})
	# xkout6 = sess.run(x6, feed_dict={y: y_test})
	# xkout7 = sess.run(x7, feed_dict={y: y_test})
	# xkout8 = sess.run(x8, feed_dict={y: y_test})
	# xkout9 = sess.run(x9, feed_dict={y: y_test})
	# xkout10 = sess.run(x10, feed_dict={y: y_test})
	# xkout11 = sess.run(x11, feed_dict={y: y_test})
	# xkout12 = sess.run(x12, feed_dict={y: y_test})
	# xkout13 = sess.run(x13, feed_dict={y: y_test})
	# xkout14 = sess.run(x14, feed_dict={y: y_test})
	# xkout15 = sess.run(x15, feed_dict={y: y_test})
	# xkout16 = sess.run(x16, feed_dict={y: y_test})
	# xkout17 = sess.run(x17, feed_dict={y: y_test})
	# xkout18 = sess.run(x18, feed_dict={y: y_test})
	# xkout19 = sess.run(x19, feed_dict={y: y_test})
	# xkout20 = sess.run(x20, feed_dict={y: y_test})

	sourcetest_mat = np.reshape(source_test, (testsymbolnum, K))
	scipy.io.savemat('data/layernum3/Hn1/5000/SNR%d/source.mat'%i1, {'source': sourcetest_mat})
	scipy.io.savemat('data/layernum3/Hn1/5000/SNR%d/x1.mat'%i1, {'x1': xkout1})
	scipy.io.savemat('data/layernum3/Hn1/5000/SNR%d/x2.mat'%i1, {'x2': xkout2})
	scipy.io.savemat('data/layernum3/Hn1/5000/SNR%d/x3.mat'%i1, {'x3': xkout3})
	# scipy.io.savemat('data/layernum3/Hn1/5000/SNR%d/x4.mat'%i1, {'x4': xkout4})
	# scipy.io.savemat('data/layernum6/Hn1/50000/SNR%d/x5.mat'%i1, {'x5': xkout5})
	# scipy.io.savemat('data/layernum6/Hn1/50000/SNR%d/x6.mat'%i1, {'x6': xkout6})
	# scipy.io.savemat('data/layernum7/Hn4/SNR%d/x7.mat'%i1, {'x7': xkout7})
	# scipy.io.savemat('data/layernum8/Hn4/SNR%d/x8.mat'%i1, {'x8': xkout8})
	# scipy.io.savemat('data/layernum20/SNR%d/x9.mat' % i1, {'x9': xkout9})
	# scipy.io.savemat('data/layernum20/SNR%d/x10.mat' % i1, {'x10': xkout10})
	# scipy.io.savemat('data/layernum20/SNR%d/x11.mat' % i1, {'x11': xkout11})
	# scipy.io.savemat('data/layernum20/SNR%d/x12.mat' % i1, {'x12': xkout12})
	# scipy.io.savemat('data/layernum20/SNR%d/x13.mat' % i1, {'x13': xkout13})
	# scipy.io.savemat('data/layernum20/SNR%d/x14.mat' % i1, {'x14': xkout14})
	# scipy.io.savemat('data/layernum20/SNR%d/x15.mat' % i1, {'x15': xkout15})
	# scipy.io.savemat('data/layernum20/SNR%d/x16.mat' % i1, {'x16': xkout16})
	# scipy.io.savemat('data/layernum20/SNR%d/x17.mat' % i1, {'x17': xkout17})
	# scipy.io.savemat('data/layernum20/SNR%d/x18.mat' % i1, {'x18': xkout18})
	# scipy.io.savemat('data/layernum20/SNR%d/x19.mat' % i1, {'x19': xkout19})
	# scipy.io.savemat('data/layernum20/SNR%d/x20.mat' % i1, {'x20': xkout20})




	scipy.io.savemat('data/layernum3/Hn1/5000/SNR%d/y.mat'%i1, {'y': y_test})




#test the model one batch
# xkout=np.zeros((K,batch_size))
# xkout=sess.run(x1,feed_dict={y:y_})
# xkout2=np.zeros((K,batch_size))
# xkout2=sess.run(x2,feed_dict={y:y_})
# xkout3=np.zeros((K,batch_size))
# xkout3=sess.run(x3,feed_dict={y:y_})
# xkout4=np.zeros((K,batch_size))
# xkout4=sess.run(x4,feed_dict={y:y_})
# source_mat= np.reshape(source, (batch_size, K))
# scipy.io.savemat('source.mat',{'source':source_mat})
# scipy.io.savemat('x1.mat',{'x1':xkout})
# scipy.io.savemat('x2.mat',{'x2':xkout2})
# scipy.io.savemat('x3.mat',{'x3':xkout3})
# scipy.io.savemat('x4.mat',{'x4':xkout4})
# scipy.io.savemat('y.mat',{'y':y_})

# #test the model
# testsymbolnum=1000000
# source_test, x_test, y_test = generate_data(testsymbolnum,K,N,H)
# xkout=np.zeros((K,testsymbolnum))
# xkout=sess.run(x1,feed_dict={y:y_test})
# xkout2=np.zeros((K,testsymbolnum))
# xkout2=sess.run(x2,feed_dict={y:y_test})
# xkout3=np.zeros((K,testsymbolnum))
# xkout3=sess.run(x3,feed_dict={y:y_test})
# xkout4=np.zeros((K,testsymbolnum))
# xkout4=sess.run(x4,feed_dict={y:y_test})
# xkout5=np.zeros((K,testsymbolnum))
# xkout5=sess.run(x5,feed_dict={y:y_test})
# xkout6=np.zeros((K,testsymbolnum))
# xkout6=sess.run(x6,feed_dict={y:y_test})
# sourcetest_mat= np.reshape(source_test, (testsymbolnum, K))
# scipy.io.savemat('source.mat',{'source':sourcetest_mat})
# scipy.io.savemat('x1.mat',{'x1':xkout})
# scipy.io.savemat('x2.mat',{'x2':xkout2})
# scipy.io.savemat('x3.mat',{'x3':xkout3})
# scipy.io.savemat('x4.mat',{'x4':xkout4})
# scipy.io.savemat('x5.mat',{'x5':xkout5})
# scipy.io.savemat('x6.mat',{'x6':xkout6})
# scipy.io.savemat('y.mat',{'y':y_test})

sess.close()
print("layernum3_Hn1_5000_16*8_end")
print("end")