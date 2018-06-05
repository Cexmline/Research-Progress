import tensorflow as tf
import numpy as np
import scipy.io


LOGDIR="path/layernum4/16x8/Hn2"
K=8  #Number of users
N=16 #Number of receiving antenna
modType=16 #bpsk:2 qpsk:4 16qam:16
modName='16qam'
batch_size=5000 #Define the batch size
STEPS=20000 #Number of iteration
Hdata=scipy.io.loadmat('Hn2_16x8_5.3765')
H=Hdata['Hn2']
Hpart1=np.concatenate((np.real(H),np.imag(H)))
Hpart2=np.concatenate(	(-np.imag(H),np.real(H)))
realH=np.concatenate((Hpart1,Hpart2),axis=1)
realH=realH.astype('float32')
#modulation
table_bpsk=np.array([1,-1]) #modType=2
table_qpsk=np.array([-0.707107+0.707107j,-0.707107-0.707107j,0.707107+0.707107j,0.707107-0.707107j])#modType=4
table_16qam= np.array([-0.948683+0.948683j, -0.948683+0.316228j, -0.948683-0.948683j, -0.948683-0.316228j, -0.316228+0.948683j, -0.316228+0.316228j, -0.316228-0.948683j, -0.316228-0.316228j,
					   0.948683+0.948683j, 0.948683+0.316228j, 0.948683-0.948683j, 0.948683-0.316228j, 0.316228+0.948683j, 0.316228+0.316228j, 0.316228-0.948683j, 0.316228-0.316228j])
modTable={'bpsk':table_bpsk,'qpsk':table_qpsk,'16qam':table_16qam}





def modulation(sourceSeq,num_Symbol,mod_Type,mod_Name):
	mod = modTable[mod_Name]
	if mod_Type==2:
		mod_Seq=np.zeros((num_Symbol,1))
	else:
		mod_Seq=np.zeros((num_Symbol,1),dtype='complex')
	if mod_Type==2:
		for i in range(num_Symbol):
			index = sourceSeq[i][0]
			mod_Seq[i] = mod[index]
			i = i + 1

	if mod_Type==4:
		for i in range(num_Symbol):
			index=sourceSeq[i*2][0]*2+sourceSeq[i*2+1][0]
			mod_Seq[i]=mod[index]
			i=i+1
	if mod_Type==16:
		for i in range (num_Symbol):
			index=sourceSeq[i*4][0]*8+sourceSeq[i*4+1][0]*4+sourceSeq[i*4+2][0]*2+sourceSeq[i*4+3][0]
			mod_Seq[i]=mod[index]
			i=i+1

	return mod_Seq
#demodulation
def demodulation(receiveSeq,num_Symbol,mod_Type):
	if mod_Type==2:
			demod_Seq=((receiveSeq<0)*1)
	if mod_Type==4:
		demod_Seq=np.zeros((num_Symbol*2,1))
		for i in range(num_Symbol):
			demod_Seq[i*2]=(receiveSeq[i].real>0)*1
			demod_Seq[i*2+1]=(receiveSeq[i].imag<0)*1
			i=i+1
	if mod_Type==16:
		demod_Seq =np.zeros((num_Symbol*4, 1))
		for i in range(num_Symbol):
			demod_Seq[i * 4] = (receiveSeq[i].real>0)*1
			demod_Seq[i * 4 + 1] = ((abs(receiveSeq[i].real)-0.632456)<0)*1
			demod_Seq[i * 4 + 2] = (receiveSeq[i].imag<0)*1
			demod_Seq[i * 4 + 3] = ((abs(receiveSeq[i].imag)-0.632456)<0)*1
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









#Define the model of the system
x=tf.placeholder(tf.float32,shape=(None,2*K),name="transmit")
y=tf.placeholder(tf.float32,shape=(None,2*N),name="receiver")

#Define the parameters of the network
#layer1
with tf.name_scope('layer1parameter'):
	w10=tf.Variable(tf.truncated_normal([6*K,6*K],stddev=1,seed=1))
	b10=tf.Variable(tf.constant(0.1,shape=[6*K,1]))
	# b10=tf.Variable(tf.zeros([5*K,1]))
	w20=tf.Variable(tf.truncated_normal([2*K,6*K],stddev=1,seed=1))
	b20=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	# w30=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	#w30=tf.Variable(tf.zeros([2*K,5*K]))
	# b30=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1",w10)
	tf.summary.histogram("weigths2",w20)
	# tf.summary.histogram("weights3",w30)
	tf.summary.histogram("biases1",b10)
	tf.summary.histogram("biases2",b20)
	# tf.summary.histogram("biases3",b30)

#layer2
with tf.name_scope('layer2parameter'):
	#w31=tf.Variable(tf.zeros([2*K,5*K]))
	w11=tf.Variable(tf.truncated_normal([6*K,6*K],stddev=1,seed=1))
	b11=tf.Variable(tf.constant(0.1,shape=[6*K,1]))
	w21=tf.Variable(tf.truncated_normal([2*K,6*K],stddev=1,seed=1))
	b21=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	# w31=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	# b31=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1",w11)
	tf.summary.histogram("weigths2",w21)
	# tf.summary.histogram("weights3",w31)
	tf.summary.histogram("biases1",b11)
	tf.summary.histogram("biases2",b21)
	# tf.summary.histogram("biases3",b31)

#layer3
with tf.name_scope('layer3parameter'):
	#w32=tf.Variable(tf.zeros([2*K,5*K]))
	w12=tf.Variable(tf.truncated_normal([6*K,6*K],stddev=1,seed=1))
	b12=tf.Variable(tf.constant(0.1,shape=[6*K,1]))
	w22=tf.Variable(tf.truncated_normal([2*K,6*K],stddev=1,seed=1))
	b22=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	# w32=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	# b32=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1", w12)
	tf.summary.histogram("weigths2", w22)
	# tf.summary.histogram("weights3", w32)
	tf.summary.histogram("biases1", b12)
	tf.summary.histogram("biases2", b22)
	# tf.summary.histogram("biases3", b32)

#layer4
with tf.name_scope('layer4parameter'):
	#w33=tf.Variable(tf.zeros([2*K,5*K]))
	w13=tf.Variable(tf.truncated_normal([6*K,6*K],stddev=1,seed=1))
	b13=tf.Variable(tf.constant(0.1,shape=[6*K,1]))
	w23=tf.Variable(tf.truncated_normal([2*K,6*K],stddev=1,seed=1))
	b23=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	# w33=tf.Variable(tf.truncated_normal([2*K,5*K],stddev=1,seed=1))
	# b33=tf.Variable(tf.constant(0.1,shape=[2*K,1]))
	tf.summary.histogram("weights1", w13)
	tf.summary.histogram("weigths2", w23)
	# tf.summary.histogram("weights3", w33)
	tf.summary.histogram("biases1", b13)
	tf.summary.histogram("biases2", b23)
	# tf.summary.histogram("biases3", b33)





#Define the process of the forward propagation 
#layer1
with tf.name_scope('layer1'):
	combination1=tf.matmul(tf.transpose(realH),tf.transpose(y))
	s=tf.shape(combination1)
	# v0=tf.zeros([2*K,s[1]])
	x0=tf.zeros([2*K,s[1]])
	HtrH=tf.matmul(tf.transpose(realH),realH)
	combination2=tf.matmul(HtrH,x0)
	concatenation=tf.concat([combination1,x0,combination2],0,name="Concatenate")
	cal1=tf.matmul(w10,concatenation)+b10
	z0=tf.nn.relu(cal1,name="Z0")
	cal2=tf.matmul(w20,z0)+b20+x0
	t0=tf.Variable(0.5,name='t0')
	x1=-1+tf.nn.relu(cal2+t0)/abs(t0)-tf.nn.relu(cal2-t0)/abs(t0)
	# v1=tf.matmul(w30,z0)+b30

#layer2
with tf.name_scope('layer2'):
	combination2_1=tf.matmul(HtrH,x1)
	concatenation_1=tf.concat([combination1,x1,combination2_1],0,name="Concatenate1")
	cal1_1=tf.matmul(w11,concatenation_1)+b11
	z1=tf.nn.relu(cal1_1,name="Z1")
	cal2_1=tf.matmul(w21,z1)+b21+x1
	t1=tf.Variable(0.5,name='t1')
	x2=-1+tf.nn.relu(cal2_1+t1)/abs(t1)-tf.nn.relu(cal2_1-t1)/abs(t1)
	# v2=tf.matmul(w31,z1)+b31

#layer3
with tf.name_scope('layer3'):
	combination2_2=tf.matmul(HtrH,x2)
	concatenation_2=tf.concat([combination1,x2,combination2_2],0,name="Concatenate2")
	cal1_2=tf.matmul(w12,concatenation_2)+b12
	z2=tf.nn.relu(cal1_2,name="Z2")
	cal2_2=tf.matmul(w22,z2)+b22+x2
	t2=tf.Variable(0.5,name='t2')
	x3=-1+tf.nn.relu(cal2_2+t2)/abs(t2)-tf.nn.relu(cal2_2-t2)/abs(t2)
	# v3=tf.matmul(w32,z2)+b32
#
#layer4
with tf.name_scope('layer4'):
	combination2_3=tf.matmul(HtrH,x3)
	concatenation_3=tf.concat([combination1,x3,combination2_3],0,name="Concatenate3")
	cal1_3=tf.matmul(w13,concatenation_3)+b13
	z3=tf.nn.relu(cal1_3,name="Z3")
	cal2_3=tf.matmul(w23,z3)+b23+x3
	t3=tf.Variable(0.5,name='t3')
	x4=-1+tf.nn.relu(cal2_3+t3)/abs(t3)-tf.nn.relu(cal2_3-t3)/abs(t3)
	# v4=tf.matmul(w33,z3)+b33


#
#

#Define loss function and backpropagation algorithm
with tf.name_scope('decorrelator'):
	xwave_part1=tf.matrix_inverse(tf.matmul(tf.transpose(realH),realH))
	xwave_part2=tf.matmul(xwave_part1,tf.transpose(realH))
	xwave=tf.matmul(xwave_part2,tf.transpose(y))

with tf.name_scope('loss'):
	lossfunction=tf.log(1.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x1))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
			 	+tf.log(2.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x2))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
             	+tf.log(3.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x3))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
             	+tf.log(4.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x4))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))
				 # +tf.log(5.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x5))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
				 # +tf.log(6.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x6))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
				 # +tf.log(7.00)*tf.reduce_sum(tf.squared_difference(tf.transpose(x),x7))/tf.reduce_sum(tf.squared_difference(tf.transpose(x),xwave))\
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
def generate_data(batchsize,K,N,H,mod_Type,mod_Name):
	symbolbits = np.log2(mod_Type)
	symbolbits =symbolbits.astype(int)
	source = np.random.randint(0, 2, ((batchsize * K )* symbolbits, 1))  # generate 0,1 bits
	x_complex = modulation(source,batchsize*K,mod_Type,mod_Name)      # QPSK modulation
	x_complex = np.reshape(x_complex, (batchsize, K))
	w = np.zeros((N, batchsize),dtype='complex')  # Noise Vector with independent,zero mean Gaussian variables of variance 1
	for m in range(batchsize):
		SNR = np.random.uniform(8, 21, 1)  ##8dB-13dB uniform distribution
		sigma = np.sqrt(1 / (10 ** (SNR / 10)))
		wpart = sigma/np.sqrt(2) * np.random.randn(N) + 1j*sigma/np.sqrt(2) * np.random.randn(N)
		w[:, m] = wpart
	# x_ = x_.astype('float32')
	y_ = np.dot(H, np.transpose(x_complex)) + w
	y_complex = np.transpose(y_)
	x_real = np.concatenate((np.real(x_complex), np.imag(x_complex)),axis=1)
	y_real = np.concatenate((np.real(y_complex), np.imag(y_complex)),axis=1)
	return source, x_complex, y_complex,x_real,y_real

def generate_testdata(symbolnum,K,N,H,mod_Type,mod_Name,SNR):
	symbolbits = np.log2(mod_Type)
	symbolbits =symbolbits.astype(int)
	np.random.seed(1)
	source = np.random.randint(0, 2, ((symbolnum * K)*symbolbits, 1))  # generate 0,1 bits
	x_complex = modulation(source, symbolnum * K, mod_Type, mod_Name)
	x_complex = np.reshape(x_complex, (symbolnum, K))
	w = np.zeros((N, symbolnum), dtype='complex')
	sigma = np.sqrt(1 / (10 ** (SNR / 10)))
	for m in range(symbolnum):
		wpart = sigma/np.sqrt(2) * np.random.randn(N) + 1j*sigma/np.sqrt(2) * np.random.randn(N)
		w[:, m] = wpart
	# x_ = x_.astype('float32')
	y_ = np.dot(H, np.transpose(x_complex)) + w
	y_complex = np.transpose(y_)
	x_real = np.concatenate((np.real(x_complex), np.imag(x_complex)),axis=1)
	y_real = np.concatenate((np.real(y_complex), np.imag(y_complex)),axis=1)
	return source, x_complex, y_complex,x_real,y_real



#Create a session to run Tensorflow
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)
writer=tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

for i in range(STEPS):
	source, x_complex,y_complex,x_, y_ = generate_data(batch_size,K,N,H,modType,modName)
	sess.run(train_step,
			feed_dict={x:x_,y:y_}) #train wk bk


	if i%500==0:
		s=sess.run(summ,feed_dict={x:x_,y:y_})
		writer.add_summary(s,i)
	if i%5000==0:
		total_lossfunction=sess.run(lossfunction,feed_dict={x:x_,y:y_})
		print("After %d training steps,cross entropy on all data is %g"%(i,total_lossfunction))
		print("x2:",sess.run(x4,feed_dict={y:y_}))
		# print("v3:",sess.run(v3,feed_dict={y:y_}))
		# print("z3:",sess.run(z3,feed_dict={y:y_}))
		print("b11:", sess.run(b11))
		print("b21:", sess.run(b21))
		# print("b31:", sess.run(b31))
		# print("b32:", sess.run(b32))
		# print("w31:", sess.run(w31))
		# print("w32:", sess.run(w32))
		print("t0:",sess.run(t0))
		print("t1:", sess.run(t1))
		print("t2:", sess.run(t2))
		print("t3:", sess.run(t3))








testsymbolnum=100000
symbolbits = np.log2(modType)
symbolbits = symbolbits.astype(int)
sourcelength = testsymbolnum * symbolbits * K
xkout1=np.zeros((2*K,testsymbolnum))
xkout2=np.zeros((2*K,testsymbolnum))
xkout3=np.zeros((2*K,testsymbolnum))
xkout4=np.zeros((2*K,testsymbolnum))
layernum=4
ber=np.zeros((layernum,13))
berdecorSet=np.zeros((1,13))
indexcount=0

for i1 in range (8,21):
	source_test,xcomplex_test,ycomplex_test,x_test,y_test=generate_testdata(testsymbolnum, K, N, H, modType, modName, i1)
	xkout1 = sess.run(x1, feed_dict={y: y_test})
	xkout2 = sess.run(x2, feed_dict={y: y_test})
	xkout3 = sess.run(x3, feed_dict={y: y_test})
	xkout4 = sess.run(x4, feed_dict={y: y_test})
	xkout1_complex=xkout1[0:K,:]+1j*xkout1[K:2*K,:]
	xkout1_complex_seq=np.reshape(np.transpose(xkout1_complex),(testsymbolnum*K,1))
	xkout2_complex=xkout2[0:K,:]+1j*xkout2[K:2*K,:]
	xkout2_complex_seq=np.reshape(np.transpose(xkout2_complex),(testsymbolnum*K,1))
	xkout3_complex=xkout3[0:K,:]+1j*xkout3[K:2*K,:]
	xkout3_complex_seq=np.reshape(np.transpose(xkout3_complex),(testsymbolnum*K,1))
	xkout4_complex=xkout4[0:K,:]+1j*xkout4[K:2*K,:]
	xkout4_complex_seq=np.reshape(np.transpose(xkout4_complex),(testsymbolnum*K,1))
	xkout1bit_seq= demodulation(xkout1_complex_seq,testsymbolnum*K,modType)
	xkout2bit_seq= demodulation(xkout2_complex_seq,testsymbolnum*K,modType)
	xkout3bit_seq = demodulation(xkout3_complex_seq, testsymbolnum * K, modType)
	xkout4bit_seq = demodulation(xkout4_complex_seq, testsymbolnum * K, modType)
	ber_layer1=np.sum(abs(xkout1bit_seq-source_test))/ sourcelength
	ber_layer2=np.sum(abs(xkout2bit_seq-source_test))/ sourcelength
	ber_layer3 = np.sum(abs(xkout3bit_seq - source_test)) / sourcelength
	ber_layer4 = np.sum(abs(xkout4bit_seq - source_test)) / sourcelength
	ber[:,indexcount]=[ber_layer1, ber_layer2, ber_layer3,ber_layer4]
	xdecor = sess.run(xwave, feed_dict={y: y_test})
	xdecor_complex=xdecor[0:K, :]+1j*xdecor[K:2*K, :]
	xdecor_complex_seq= np.reshape(np.transpose(xdecor_complex),(testsymbolnum*K,1))
	xdecor_seq= demodulation(xdecor_complex_seq,testsymbolnum*K,modType)
	ber_decor= np.sum(abs(xdecor_seq-source_test))/ sourcelength
	berdecorSet[:,indexcount]=ber_decor
	indexcount=indexcount+1
print("ber:",ber)
print("ber_decorSet",berdecorSet)
scipy.io.savemat('result/layernum4/16x8/Hn2/ber.mat',{'ber':ber})
scipy.io.savemat('result/layernum4/16x8/Hn2/x4.mat', {'x4': xkout4bit_seq})
scipy.io.savemat('result/layernum4/16x8/Hn2/source.mat', {'source': source_test})
scipy.io.savemat('result/layernum4/16x8/Hn2/berdecorSet.mat', {'berdecorSet': berdecorSet})

sess.close()
print("layernum4_Hn2_5000_16x8_end")
print("end")