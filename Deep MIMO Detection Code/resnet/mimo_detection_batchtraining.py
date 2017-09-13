import tensorflow as tf
import numpy as np
import scipy.io
K=4  #Number of users
N=8 #Number of receiving antenna
batch_size=5000 #Define the batch size 
STEPS=100000  #Number of iteration
Hdata=scipy.io.loadmat('H_8*4')
H=Hdata['H']
H=H.astype('float32')

#Define the model of the system
x=tf.placeholder(tf.float32,shape=(None,K),name="transmit")
y=tf.placeholder(tf.float32,shape=(None,N),name="receiver")

#Define the parameters of the network
#layer1
w10=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b10=tf.Variable(tf.zeros([5*K,1]))
w20=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b20=tf.Variable(tf.zeros([K,1]))
w30=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b30=tf.Variable(tf.zeros([2*K,1]))

#layer2
w11=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b11=tf.Variable(tf.zeros([5*K,1]))
w21=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b21=tf.Variable(tf.zeros([K,1]))
w31=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b31=tf.Variable(tf.zeros([2*K,1]))

#layer3
w12=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b12=tf.Variable(tf.zeros([5*K,1]))
w22=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b22=tf.Variable(tf.zeros([K,1]))
w32=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b32=tf.Variable(tf.zeros([2*K,1]))

#layer4
w13=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b13=tf.Variable(tf.zeros([5*K,1]))
w23=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b23=tf.Variable(tf.zeros([K,1]))
w33=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b33=tf.Variable(tf.zeros([2*K,1]))

#layer5
w14=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b14=tf.Variable(tf.zeros([5*K,1]))
w24=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b24=tf.Variable(tf.zeros([K,1]))
w34=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b34=tf.Variable(tf.zeros([2*K,1]))

#layer6
w15=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b15=tf.Variable(tf.zeros([5*K,1]))
w25=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b25=tf.Variable(tf.zeros([K,1]))
w35=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b35=tf.Variable(tf.zeros([2*K,1]))

#layer7
w16=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b16=tf.Variable(tf.zeros([5*K,1]))
w26=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b26=tf.Variable(tf.zeros([K,1]))
w36=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b36=tf.Variable(tf.zeros([2*K,1]))

#layer8
w17=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b17=tf.Variable(tf.zeros([5*K,1]))
w27=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b27=tf.Variable(tf.zeros([K,1]))
w37=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b37=tf.Variable(tf.zeros([2*K,1]))





#Define the process of the forward propagation 
#layer1
combination1=tf.matmul(tf.transpose(H),tf.transpose(y))
s=tf.shape(combination1)
v0=tf.zeros([2*K,s[1]])
x0=tf.zeros([K,s[1]])
HtrH=tf.matmul(tf.transpose(H),H)
combination2=tf.matmul(HtrH,x0)
concatenation=tf.concat([combination1,x0,combination2,v0],0,name="Concatenate")
cal1=tf.matmul(w10,concatenation)+b10
z0=tf.nn.relu(cal1,name="Z0")
cal2=tf.matmul(w20,z0)+b20
t0=0.5
x1=-1+tf.nn.relu(cal2+t0)/abs(t0)-tf.nn.relu(cal2-t0)/abs(t0)
v1=tf.matmul(w30,z0)+b30

#layer2
combination2_1=tf.matmul(HtrH,x1)
concatenation_1=tf.concat([combination1,x1,combination2_1,v1],0,name="Concatenate1")
cal1_1=tf.matmul(w11,concatenation_1)+b11
z1=tf.nn.relu(cal1_1,name="Z1")
cal2_1=tf.matmul(w21,z1)+b21+x1
t1=0.5
x2=-1+tf.nn.relu(cal2_1+t1)/abs(t1)-tf.nn.relu(cal2_1-t1)/abs(t1)
v2=tf.matmul(w31,z1)+b31

#layer3
combination2_2=tf.matmul(HtrH,x2)
concatenation_2=tf.concat([combination1,x2,combination2_2,v2],0,name="Concatenate2")
cal1_2=tf.matmul(w12,concatenation_2)+b12
z2=tf.nn.relu(cal1_2,name="Z2")
cal2_2=tf.matmul(w22,z2)+b22+x2
t2=0.5
x3=-1+tf.nn.relu(cal2_2+t2)/abs(t2)-tf.nn.relu(cal2_2-t2)/abs(t2)
v3=tf.matmul(w32,z2)+b32

#layer4
combination2_3=tf.matmul(HtrH,x3)
concatenation_3=tf.concat([combination1,x3,combination2_3,v3],0,name="Concatenate3")
cal1_3=tf.matmul(w13,concatenation_3)+b13
z3=tf.nn.relu(cal1_3,name="Z3")
cal2_3=tf.matmul(w23,z3)+b23+x3
t3=0.5
x4=-1+tf.nn.relu(cal2_3+t3)/abs(t3)-tf.nn.relu(cal2_3-t3)/abs(t3)
v4=tf.matmul(w33,z3)+b33

#layer5
combination2_4=tf.matmul(HtrH,x4)
concatenation_4=tf.concat([combination1,x4,combination2_4,v4],0,name="Concatenate4")
cal1_4=tf.matmul(w14,concatenation_4)+b14
z4=tf.nn.relu(cal1_4,name="Z4")
cal2_4=tf.matmul(w24,z4)+b24+x4
t4=0.5
x5=-1+tf.nn.relu(cal2_4+t4)/abs(t4)-tf.nn.relu(cal2_4-t4)/abs(t4)
v5=tf.matmul(w34,z4)+b34

#layer6
combination2_5=tf.matmul(HtrH,x5)
concatenation_5=tf.concat([combination1,x5,combination2_5,v5],0,name="Concatenate5")
cal1_5=tf.matmul(w15,concatenation_5)+b15
z5=tf.nn.relu(cal1_5,name="Z5")
cal2_5=tf.matmul(w25,z5)+b25+x5
t5=0.5
x6=-1+tf.nn.relu(cal2_5+t5)/abs(t5)-tf.nn.relu(cal2_5-t5)/abs(t5)
v6=tf.matmul(w35,z5)+b35

#layer7
combination2_6=tf.matmul(HtrH,x6)
concatenation_6=tf.concat([combination1,x6,combination2_6,v6],0,name="Concatenate6")
cal1_6=tf.matmul(w16,concatenation_6)+b16
z6=tf.nn.relu(cal1_6,name="Z6")
cal2_6=tf.matmul(w26,z6)+b26+x6
t6=0.5
x7=-1+tf.nn.relu(cal2_6+t6)/abs(t6)-tf.nn.relu(cal2_6-t6)/abs(t6)
v7=tf.matmul(w36,z6)+b36

#layer8
combination2_7=tf.matmul(HtrH,x7)
concatenation_7=tf.concat([combination1,x7,combination2_7,v7],0,name="Concatenate7")
cal1_7=tf.matmul(w17,concatenation_7)+b17
z7=tf.nn.relu(cal1_7,name="Z7")
cal2_7=tf.matmul(w27,z7)+b27+x7
t7=0.5
x8=-1+tf.nn.relu(cal2_7+t7)/abs(t7)-tf.nn.relu(cal2_7-t7)/abs(t7)
v8=tf.matmul(w37,z7)+b37





#Define loss function and backpropagation algorithm
xwave_part1=tf.matrix_inverse(tf.matmul(tf.transpose(H),H))
xwave_part2=tf.matmul(xwave_part1,tf.transpose(H))
xwave=tf.matmul(xwave_part2,tf.transpose(y))

lossfunction=tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x1))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
			 +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x2))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x3))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x4))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x5))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
			 +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x6))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x7))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x8))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))
train_step=tf.train.AdamOptimizer(0.001).minimize(lossfunction)




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
for i in range(STEPS):
	source, x_, y_ = generate_data(batch_size,K,N,H)
	sess.run(train_step,
			feed_dict={x:x_,y:y_}) #train wk bk

	if i%5000==0:
		total_lossfunction=sess.run(lossfunction,feed_dict={x:x_,y:y_})
		print("After %d training steps,cross entropy on all data is %g"%(i,total_lossfunction))
		print("x3:",sess.run(x3,feed_dict={y:y_}))
		print("v3:",sess.run(v3,feed_dict={y:y_}))
		print("z3:",sess.run(z3,feed_dict={y:y_}))



print("w11:",sess.run(w11))
print("w21:",sess.run(w21))
print("w31:",sess.run(w31))
print("b11:",sess.run(b11))
print("b21:",sess.run(b21))
print("b31:",sess.run(b31))



testsymbolnum=100000
xkout=np.zeros((K,testsymbolnum))
xkout2=np.zeros((K,testsymbolnum))
xkout3=np.zeros((K,testsymbolnum))
xkout4=np.zeros((K,testsymbolnum))
xkout5=np.zeros((K,testsymbolnum))
xkout6=np.zeros((K,testsymbolnum))
xkout7=np.zeros((K,testsymbolnum))
xkout8=np.zeros((K,testsymbolnum))
for i1 in range (8,14):
	source_test,x_test,y_test=generate_testdata(testsymbolnum, K, N, H, i1)
	xkout1 = sess.run(x1, feed_dict={y: y_test})
	xkout2 = sess.run(x2, feed_dict={y: y_test})
	xkout3 = sess.run(x3, feed_dict={y: y_test})
	xkout4 = sess.run(x4, feed_dict={y: y_test})
	xkout5 = sess.run(x5, feed_dict={y: y_test})
	xkout6 = sess.run(x6, feed_dict={y: y_test})
	xkout7 = sess.run(x7, feed_dict={y: y_test})
	xkout8 = sess.run(x8, feed_dict={y: y_test})
	sourcetest_mat = np.reshape(source_test, (testsymbolnum, K))
	scipy.io.savemat('data/layernum8/SNR%d/source.mat'%i1, {'source': sourcetest_mat})
	scipy.io.savemat('data/layernum8/SNR%d/x1.mat'%i1, {'x1': xkout1})
	scipy.io.savemat('data/layernum8/SNR%d/x2.mat'%i1, {'x2': xkout2})
	scipy.io.savemat('data/layernum8/SNR%d/x3.mat'%i1, {'x3': xkout3})
	scipy.io.savemat('data/layernum8/SNR%d/x4.mat'%i1, {'x4': xkout4})
	scipy.io.savemat('data/layernum8/SNR%d/x5.mat'%i1, {'x5': xkout5})
	scipy.io.savemat('data/layernum8/SNR%d/x6.mat'%i1, {'x6': xkout6})
	scipy.io.savemat('data/layernum8/SNR%d/x7.mat'%i1, {'x7': xkout7})
	scipy.io.savemat('data/layernum8/SNR%d/x8.mat'%i1, {'x8': xkout8})
	scipy.io.savemat('data/layernum8/SNR%d/y.mat'%i1, {'y': y_test})




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
print("end")