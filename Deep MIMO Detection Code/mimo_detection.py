import tensorflow as tf
import numpy as np
import scipy.io
K=10  #Number of users
N=20  #Number of receiving antenna
batch_size=5000 #Define the batch size 
STEPS=50000  #Number of iteration
Hdata=scipy.io.loadmat('H_20*10')
H=Hdata['H']
H=H.astype('float32')


#H=np.array([],dtype=np.float32)   #N*K

#Define the model of the system
dataset_size=batch_size*500
source=np.random.randint(0,2,(dataset_size*K,1)) #generate 0,1 bits
x_=-2.0*source+1.0      #BPSK modulation 
x_=np.reshape(x_,(K,batch_size*500))
w=np.zeros((N,batch_size*500)) #Noise Vector with independent,zero mean Gaussian variables of variance 1
for j in range (batch_size*500):
	wpart=np.random.randn(N)
	w[:,j]=wpart
x_=x_.astype('float32')
y_=np.dot(H,x_)+w

#H=tf.placeholder(tf.float32,shape=(N,K),name="ChannelMatrix")
x=tf.placeholder(tf.float32,shape=(K,None),name="transmit")
y=tf.placeholder(tf.float32,shape=(N,None),name="receiver")


#Define the parameters of the network
#init_w1k=np.random.randn(5*K,5*K)
#init_wik_placeholder=tf.placeholder(tf.float32,shape=init_w1k.shape)
#w1k=tf.Variable(init_wik_placeholder)
w1k=tf.Variable(tf.random_normal([5*K,5*K],stddev=1,seed=1))
b1k=tf.Variable(tf.zeros([5*K,batch_size]))
#b1k=tf.Variable(tf.random_normal([5*K,batch_size],stddev=1,seed=1))
w2k=tf.Variable(tf.random_normal([K,5*K],stddev=1,seed=1))
b2k=tf.Variable(tf.zeros([K,batch_size]))
#b2k=tf.Variable(tf.random_normal([K,batch_size],stddev=1,seed=1))
w3k=tf.Variable(tf.random_normal([2*K,5*K],stddev=1,seed=1))
b3k=tf.Variable(tf.zeros([2*K,batch_size]))
#b3k=tf.Variable(tf.random_normal([2*K,batch_size],stddev=1,seed=1))

#Define the process of the forward propagation 
combination1=tf.matmul(tf.transpose(H),y)

vk=tf.zeros([2*K,batch_size])
xk=tf.zeros([K,batch_size])

combination2=tf.matmul(tf.matmul(tf.transpose(H),H),xk)
concatenation=tf.concat([combination1,xk,combination2,vk],0,name="Concatenate")

Cal1=tf.matmul(w1k,concatenation)+b1k
zk=tf.nn.relu(Cal1,name="Zk")

cal2=tf.matmul(w2k,zk)+b2k
t=0.5
xk=-1+tf.nn.relu(cal2+t)/abs(t)-tf.nn.relu(cal2-t)/abs(t)

vk=tf.matmul(w3k,zk)+b3k

#Define loss function and backpropagation algorithm
xwave_part1=tf.matrix_inverse(tf.matmul(tf.transpose(H),H))
xwave_part2=tf.matmul(xwave_part1,tf.transpose(H))
xwave=tf.matmul(xwave_part2,y)
lossfunction=tf.log(tf.reduce_sum(tf.squared_difference(x,xk))/tf.reduce_sum(tf.squared_difference(x,xwave)))
train_step=tf.train.AdamOptimizer(0.001).minimize(lossfunction)

#Create a session to run Tensorflow
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)


print(sess.run(w1k))
print(sess.run(w2k))
print(sess.run(w3k))
print(sess.run(b1k))
print(sess.run(b2k))
print(sess.run(b3k))

for i in range(STEPS):
	start=(i*batch_size)%dataset_size
	end =min(start+batch_size,dataset_size)
	sess.run(train_step,
			feed_dict={x:x_[:,start:end],y:y_[:,start:end]})
	if i%1000==0:
		total_lossfunction=sess.run(lossfunction,feed_dict={x:x_[:,start:end],y:y_[:,start:end]})
		print("After %d training steps,cross entropy on all data is %g"%(i,total_lossfunction))
		print("xk:",sess.run(xk,feed_dict={x:x_[:,start:end],y:y_[:,start:end]}))
		print("vk:",sess.run(vk,feed_dict={x:x_[:,start:end],y:y_[:,start:end]}))

print("w1k:",sess.run(w1k))
print("w2k:",sess.run(w2k))
print("w3k:",sess.run(w3k))
print("b1k:",sess.run(b1k))
print("b2k:",sess.run(b2k))
print("b3k:",sess.run(b3k))

sess.close()

