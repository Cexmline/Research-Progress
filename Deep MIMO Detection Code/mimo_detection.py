import tensorflow as tf
import numpy as np
K=2   #Number of users
N=4   #Number of receiving antenna
#Define the batch size and channel matrix H_
batch_size=500
STEPS=500  #Number of iteration

#H_=???????

#Define the model of the system
dataset_size=batch_size*K*STEPS
source=np.random.randint(0,2,(dataset_size,1))
x=-2*source+1
x=np.reshape(x,(K,batch_size*STEPS))
w=tf.random_normal([N,1],mean=0,stddev=1) #Noise Vector with independent,zero mean Gaussian variables of variance 1
y_=tf.matmul(H,x)+w



x=tf.placeholder(tf.float32,shape=(K,batch_size),name="transmit")
y=tf.placeholder(tf.float32,shape=(N,batch_size),name="receiver")
#H=tf.placeholder(tf.float32,shape=(N,K),name="ChannelMatrix")


#Define the parameters of the network
w1k=tf.Variable(tf.random_normal([8*K,5*K],stddev=1,seed=1))
b1k=tf.Variable(tf.random_normal([8*K,1],stddev=1,seed=1))
w2k=tf.Variable(tf.random_normal([K,8*K],stddev=1,seed=1))
b2k=tf.Variable(tf.random_normal([K,1],stddev=1,seed=1))
w3k=tf.Variable(tf.random_normal([2*K,8*K],stddev=1,seed=1))
b3k=tf.Variable(tf.random_normal([2*K,1],stddev=1,seed=1))

#Define the process of the forward propagation 
combination1=tf.matmul(tf.transpose(H),y)

vk=tf.zeros([2*K,1])
xk=tf.zeros([K,1])


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

for i in range(STEPS):
	start=(i*batch_size)%dataset_size
	end =min(start+batch_size,dataset_size)

	sess.run(train_step,
			feed_dict={x:X[start:end],y:y_[]})
	if i%100==0:
		total_lossfunction=sess.run(
			lossfunction,feed_dict{x:X,y:y_})
		print("After %d training steps,cross entropy on all data is %g"%(i,total_lossfunction))




sess.close()
