import tensorflow as tf
import numpy as np
import scipy.io
K=30  #Number of users
N=60  #Number of receiving antenna
batch_size=5000 #Define the batch size 
STEPS=50000  #Number of iteration
Hdata=scipy.io.loadmat('H_60*30')
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




#Define the process of the forward propagation 
#layer1
combination1=tf.matmul(tf.transpose(H),tf.transpose(y))
v0=tf.zeros([2*K,batch_size])
x0=tf.zeros([K,batch_size])
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
cal2_1=tf.matmul(w21,z1)+b21
t1=0.5
x2=-1+tf.nn.relu(cal2_1+t1)/abs(t1)-tf.nn.relu(cal2_1-t1)/abs(t1)
v2=tf.matmul(w31,z1)+b31

#layer3
combination2_2=tf.matmul(HtrH,x2)
concatenation_2=tf.concat([combination1,x2,combination2_2,v2],0,name="Concatenate2")
cal1_2=tf.matmul(w12,concatenation_2)+b12
z2=tf.nn.relu(cal1_2,name="Z2")
cal2_2=tf.matmul(w22,z2)+b22
t2=0.5
x3=-1+tf.nn.relu(cal2_2+t2)/abs(t2)-tf.nn.relu(cal2_2-t2)/abs(t2)
v3=tf.matmul(w32,z2)+b32

#layer4
combination2_3=tf.matmul(HtrH,x3)
concatenation_3=tf.concat([combination1,x3,combination2_3,v3],0,name="Concatenate3")
cal1_3=tf.matmul(w13,concatenation_3)+b13
z3=tf.nn.relu(cal1_3,name="Z3")
cal2_3=tf.matmul(w23,z3)+b23
t3=0.5
x4=-1+tf.nn.relu(cal2_3+t3)/abs(t3)-tf.nn.relu(cal2_3-t3)/abs(t3)
v4=tf.matmul(w33,z3)+b33

#layer5
combination2_4=tf.matmul(HtrH,x4)
concatenation_4=tf.concat([combination1,x4,combination2_4,v4],0,name="Concatenate3")
cal1_4=tf.matmul(w14,concatenation_4)+b14
z4=tf.nn.relu(cal1_4,name="Z4")
cal2_4=tf.matmul(w24,z4)+b24
t4=0.5
x5=-1+tf.nn.relu(cal2_4+t4)/abs(t4)-tf.nn.relu(cal2_4-t4)/abs(t4)
v5=tf.matmul(w34,z4)+b34



#Define loss function and backpropagation algorithm
xwave_part1=tf.matrix_inverse(tf.matmul(tf.transpose(H),H))
xwave_part2=tf.matmul(xwave_part1,tf.transpose(H))
xwave=tf.matmul(xwave_part2,tf.transpose(y))

lossfunction=tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x1))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
			 +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x2))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x3))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x4))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))\
             +tf.log(tf.reduce_sum(tf.squared_difference(tf.transpose(x),x5))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave)))
train_step=tf.train.AdamOptimizer(0.001).minimize(lossfunction)

#Create a session to run Tensorflow
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)


# print(sess.run(w10))
# print(sess.run(w20))
# print(sess.run(w30))
# print(sess.run(b10))
# print(sess.run(b20))
# print(sess.run(b30))

for i in range(STEPS):
	source=np.random.randint(0,2,(batch_size*K,1)) #generate 0,1 bits
	source_mat=np.reshape(source,(batch_size,K))
	x_=-2.0*source+1.0      #BPSK modulation 
	x_=np.reshape(x_,(batch_size,K))
	w=np.zeros((N,batch_size)) #Noise Vector with independent,zero mean Gaussian variables of variance 1
	for j in range (batch_size):
		wpart=0.2239*np.random.randn(N)
		w[:,j]=wpart
	x_=x_.astype('float32')
	y_=np.dot(H,np.transpose(x_))+w
	y_=np.transpose(y_)


	sess.run(train_step,
			feed_dict={x:x_,y:y_}) #train wk bk
	if i%5000==0:
		total_lossfunction=sess.run(lossfunction,feed_dict={x:x_,y:y_})
		print("After %d training steps,cross entropy on all data is %g"%(i,total_lossfunction))
		print("x5:",sess.run(x5,feed_dict={y:y_}))
		print("v5:",sess.run(v5,feed_dict={y:y_}))
		print("z4:",sess.run(z4,feed_dict={y:y_}))



print("w13:",sess.run(w10))
print("w23:",sess.run(w20))
print("w33:",sess.run(w30))
print("b13:",sess.run(b10))
print("b23:",sess.run(b20))
print("b33:",sess.run(b30))
print("w14:",sess.run(w11))
print("w24:",sess.run(w21))
print("w34:",sess.run(w31))
print("b14:",sess.run(b11))
print("b24:",sess.run(b21))
print("b34:",sess.run(b31))

xkout=np.zeros((K,batch_size))
xkout=sess.run(x5,feed_dict={y:y_})
scipy.io.savemat('source.mat',{'source':source_mat})
scipy.io.savemat('x5.mat',{'x5':xkout})

sess.close()
print("end")