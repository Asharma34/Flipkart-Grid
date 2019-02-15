import tensorflow as tf
import take_input as input_
import model_arcitecture as model               
import os
import numpy as np 
import time
import pandas as pd

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 20, "training epoch")
tf.app.flags.DEFINE_float('test_size',0.05,'test size')
tf.app.flags.DEFINE_integer('batch_size', 4, "batch size")
tf.app.flags.DEFINE_float('val_size',0.01,'val size')
tf.app.flags.DEFINE_boolean('train',True,'training')
tf.app.flags.DEFINE_float('learning_rate_',0.05,'learning rate')
tf.app.flags.DEFINE_float('keep_prob',0.8,'keep prob')
tf.app.flags.DEFINE_string('save_name','saved_model','folder of saving')
tf.app.flags.DEFINE_integer('validation_interval',1000,'validation_interval')



def train_():
	# take input data from take_input
	train_csv=pd.read_csv('training.csv')
	train,val,test=input_.make_val_test(train_csv,val_frac=FLAGS.val_size,test_frac=FLAGS.test_size)
	print('train size :: {}'.format(train.shape[0]))
	print('val size :: {}'.format(val.shape[0]))
	print('test size :: {}'.format(test.shape[0]))

	# let's have the input placeholders
	X,y,learning_rate,dropout_keep_prob,training=model.placeholders(480,640,3,4)   
	preds=model.network(X,keep_prob=dropout_keep_prob,label_cnt=4,training=training)
	loss=model.loss(preds,y)
	iou=model.iou_score(preds,y)
	optimizer=model.optimizer(loss,learning_rate)

	init=tf.global_variables_initializer()
	sess=tf.Session()
	sess.run(init)

	
	merged=tf.summary.merge_all()
	writer_train_addr='./summary/train'
	writer_val_addr='./summary/val'
	train_writer=tf.summary.FileWriter(writer_train_addr,sess.graph)
	val_writer=tf.summary.FileWriter(writer_val_addr,sess.graph)

	saver=tf.train.Saver()
	saver_addr=os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt')
	if not os.path.isdir(os.path.join(os.getcwd(),FLAGS.save_name)):
		os.mkdir(FLAGS.save_name)

	if os.path.isdir(os.path.join(os.getcwd(),FLAGS.save_name)):
		print('restoring model...')
		saver.restore(sess,os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt'))
	
	train_num=train.shape[0]
	batch_size=FLAGS.batch_size
	num_batches=int(np.ceil(train_num/batch_size))

	lr=FLAGS.learning_rate_
	kp=FLAGS.keep_prob
	epochs=FLAGS.epoch
	is_train=FLAGS.train 
	
	global_step=0
	for epoch in range(epochs):
		if  epoch > 0:
			lr /= 1.414
		batches=input_.batch(train,batch_size)
		i=0
		epoch_loss=0
		for batch in range(num_batches):
			train_x,train_y=next(batches)

			if global_step%20==0:
				summary,_,batch_loss,iou_score=sess.run([merged,optimizer,loss,iou],feed_dict={X:train_x,y:train_y,learning_rate:lr,dropout_keep_prob:kp,training:True})
				train_writer.add_summary(summary, global_step/20)
			else:
				_,batch_loss,iou_score=sess.run([optimizer,loss,iou],feed_dict={X:train_x,y:train_y,learning_rate:lr,dropout_keep_prob:kp,training:True})

			epoch_loss+=batch_loss
			print('>> t_loss - {} \t  iou : {} \t  step {} epoch {} on {} images from {} to {} out of {}  with lr {}'.format(batch_loss,iou_score,i,epoch,train_x.shape[0],i*batch_size,(i+1)*batch_size,train_num,lr))

			'''
			if i%FLAGS.validation_interval==0 and i>0:
				#validation_start_time=time.time()
				val_loss=0
				val_batch=8
				for v in range(0,val.shape[0],val_batch):
					end=v+val_batch
					if v+val_batch>val.shape[0]:
						end=val.shape[0]
					val_images,val_labels=input_.make_batch(val.iloc[v:end,:])
					print(val_images.shape)
					print(val_labels.shape)
					batch_loss,summary=sess.run([loss,merged],feed_dict={X:val_images,y:val_labels,dropout_keep_prob:1.0,training:False})
					val_writer.add_summary(summary)
					val_loss+=batch_loss
				print('>> validation loss computed :: {} on {} images'.format((val_loss/(int(np.ceil(val_images.shape[0]/32)))),val_images.shape[0]))
			'''
			
			i+=1
			global_step+=1

			'''
			#inds=[1,2,3]
		#sample_x=train_x[inds]
		#sample_y=train_y[inds]
		#out_preds=sess.run(preds,feed_dict={X:sample_x,dropout_keep_prob:1.0,training:False})
		'''
		print('>> epoch loss computed :: {} '.format(epoch_loss/num_batches))
		#input_.show(sample_x,out_preds)
		saver.save(sess, saver_addr)
	train_writer.close()
	val_writer.close()

	sess.close()

	#test_(test)

def test_(test_images,test_labels):
	tf.reset_default_graph()
	sess=tf.Session()

	new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt.meta'))
	new_saver.restore(sess,os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt'))
	print('graph restored')
	ops=sess.graph.get_operations()
	for x in ops:
		print(x.name)

	train_csv=pd.read_csv('training.csv')

	

	_,_,test_images=input_.make_val_test(train_csv,val_frac=FLAGS.val_size,test_frac=FLAGS.test_size)
	test_images=test_images.sample(frac=1)
	test_images,test_labels=input_.make_batch(test_images.iloc[:15,:])

	print(test_images.shape)
		

	sess_input=sess.graph.get_tensor_by_name('placeholder/image:0')
	sess_out=sess.graph.get_tensor_by_name('fc1layer/BiasAdd:0')
	sess_keep_prob=sess.graph.get_tensor_by_name('placeholder/keep_prob:0')
	sess_training=sess.graph.get_tensor_by_name('placeholder/is_training:0')               

	out=sess.run(sess_out,feed_dict={sess_input:test_images,sess_keep_prob:1.0,sess_training:False})    

	print(out.shape)
	#print()
	input_.show(test_images,out,test_labels)
	sess.close()

def main(_):
	if FLAGS.train:
		print('training')
		train_()
	else:
		test_(None,None)

if __name__=='__main__':
	tf.app.run()