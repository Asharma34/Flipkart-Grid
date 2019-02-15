#architecture.py

import tensorflow as tf
import ops as op

def placeholders(img_1=480,img_2=640,channels=3,label_cnt=4):
	with tf.variable_scope('placeholder'):
		input_img=tf.placeholder(shape=[None,img_1,img_2,channels],dtype=tf.float32,name='image')
		labels=tf.placeholder(shape=[None,label_cnt],dtype=tf.float32,name='target')
		learning_rate=tf.placeholder(shape=None,dtype=tf.float32,name='learning_rate')
		keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
		training=tf.placeholder(shape=None,dtype=tf.bool,name='is_training')
		return input_img,labels,learning_rate,keep_prob,training
		
def network(X,training,keep_prob,label_cnt):

	print('input shape {}'.format(X.get_shape().as_list()))

	with tf.variable_scope('conv1layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[6,8],out_channels=32,stride_size=1,padding='SAME',a=None)
		X=tf.nn.max_pool(X,ksize=[1,6,8,1],strides=[1,6,8,1],padding='VALID')
	
	print('conv1layer',X.get_shape().as_list())

	with tf.variable_scope('conv2layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=64,stride_size=1,padding='SAME',a=None)
		X=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		X1=X

	print('conv2layer',X.get_shape().as_list())
	print('convX1layer initial',X1.get_shape().as_list())


	with tf.variable_scope('conv3layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=128,stride_size=1,padding='SAME')
		X=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		X2=X
	print('conv3layer',X.get_shape().as_list())
	print('convX2layer initial',X2.get_shape().as_list())


	with tf.variable_scope('conv4layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=128,stride_size=1,padding='SAME')
		X=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		X3=X
		print('conv3layer',X.get_shape().as_list())
		print('convX3layer initial',X3.get_shape().as_list())

	with tf.variable_scope('conv5layer'):
		X=op.batch_norm(X,training=training)
		X=tf.nn.relu(X)
		X=op.conv(X,filter_size=[3,3],out_channels=128,stride_size=1,padding='SAME')
		X=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	print('conv4layer Xres',X.get_shape().as_list())

	with tf.variable_scope('pooling_sum'):
		X1=op.avgpool(X1,filter_size=40,stride_size=40,padding='VALID')
		print('X1 final shape {}'.format(X1.get_shape().as_list()))
		X2=op.avgpool(X2,filter_size=20,stride_size=20,padding='VALID')
		print('X2 final shape {}'.format(X2.get_shape().as_list()))
		X3=op.avgpool(X3,filter_size=10,stride_size=10,padding='VALID')
		print('X3 final shape {}'.format(X3.get_shape().as_list()))
		X=op.avgpool(X,filter_size=5,stride_size=5,padding='VALID')
		print('X final shape {}'.format(X.get_shape().as_list()))


	with tf.variable_scope('concat_layers'):
		Xconcat=tf.concat([X,X1,X2,X3],name='concat_op',axis=3)

	print('concatlayer',Xconcat.get_shape().as_list())


	with tf.variable_scope('fc1layer'):
		Xfinal=op.batch_norm(Xconcat,training)
		Xfinal=op.fc(Xfinal,output_size=label_cnt,a=None)

	print('finallayer',Xfinal.get_shape().as_list())

	
	return Xfinal

def loss(preds,target):
	with tf.variable_scope('loss'):
		loss=tf.divide(tf.reduce_sum(tf.square(tf.subtract(preds,target,name='difference'),name='square'),name='sum_of_all'),16.0,name='divide_4')
	tf.summary.scalar('loss',loss)
	return loss

def iou_score(preds,labels):
	with tf.variable_scope('Iou_score'):
		x11, y11, x12, y12 = tf.split(preds, 4, axis=1)
		x21, y21, x22, y22 = tf.split(labels, 4, axis=1)
		xA = tf.maximum(x11, x21)
		yA = tf.maximum(y11, y21)
		xB = tf.minimum(x12, x22)
		yB = tf.minimum(y12, y22)
		interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB - yA + 1), 0)
		boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
		boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

		iou_= interArea / (boxAArea + boxBArea - interArea)
		iou=tf.reduce_mean(iou_)

	tf.summary.scalar('Iou_score',iou)
	return iou


def optimizer(loss,learning_rate):
	with tf.variable_scope('AdamOptimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)
	return train_op


