
import tensorflow as tf 
import deepGW
import numpy as np 
import time 
import scipy.io as sio

lr = 0.001
snr = 0.2
train_step_size = 256
test_step_size = 256
num_gpus = 1
log_device_placement = False
num_step = 10000

def tower_loss(scope, inputs, labels):

	inputs = deepGW.convert_to_tensor_float(inputs)
	labels = deepGW.convert_to_tensor_float(labels)

	pred, logits = deepGW.inference_4conv(inputs)

	_ = deepGW.loss(logits, labels)

	_ = deepGW.accuracy(pred, labels)

	losses = tf.get_collection('losses', scope)
	accuracies = tf.get_collection('accuracies', scope)

	total_loss = tf.add_n(losses, name='total_loss')

	total_accuracy = tf.add_n(accuracies, name='total_accuracy')

	return total_loss, total_accuracy


# borrowed online from tensorflow.org 
def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads 


def train(inputs):
	with tf.device('/cpu:0'):
		global_step = tf.Variable(0, name='global_step', trainable=False)

		opt = tf.train.AdamOptimizer(lr)

		with tf.name_scope('Input'):
			X = tf.placeholder(tf.float32, [None, num_gpus, 8192])

		with tf.name_scope('Label'):
			Y_ = tf.placeholder(tf.float32, [None, num_gpus, 2])

		tower_grads = []

		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(num_gpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('tower_%d' % i) as scope:
					
						loss, accuracy = tower_loss(scope, X[:, i, :], Y_[:, i, :])
						tf.get_variable_scope().reuse_variables()

						grads = opt.compute_gradients(loss)

						tower_grads.append(grads)

		grads = average_gradients(tower_grads)

		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		saver = tf.train.Saver()

		init = tf.global_variables_initializer()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        		log_device_placement=log_device_placement))
    
		sess.run(init)

		tf.train.start_queue_runners(sess=sess)

		train_error = []
		train_acc = []
		# for epoch in range(num_epoch):
			
			# num_step = len(inputs) // (train_step_size * num_gpus)
			
			# input_epoch, label_epoch = deepGW.prepare_data(inputs, labels)
			# input_epoch, label_epoch = deepGW.generate_batch_input(inputs, 'train', snr, train_step_size*num_gpus)
		for step in range(num_step):
			
			input_epoch, label_epoch = deepGW.generate_batch_input(inputs, 'train', snr, num_gpus*train_step_size, num_step, step)

			input_batch, label_batch = deepGW.get_a_batch(input_epoch, label_epoch, train_step_size, num_gpus, 0)

			start_time = time.time()

			_, loss_value, acc_value = sess.run([apply_gradient_op, loss, accuracy], feed_dict={X: input_batch, Y_: label_batch})

			duration = time.time() - start_time

			train_error.append(loss_value)
			train_acc.append(acc_value)

			if step % 10 == 0:

				num_examples_per_step = train_step_size * num_gpus
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration / num_gpus

				format_str = ('step %d, loss = %.5f, accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)')

				print(format_str % (step, loss_value, acc_value, examples_per_sec, sec_per_batch))

			
			if step % 1000 == 0 or (step + 1) == num_step:
				saver.save(sess, '/home/ruilan2/scratch/save_proj_1gpu.ckpt', global_step=step)
			

		print("<<<Training Finished!>>>")
		sio.savemat('/home/ruilan2/scratch/train_cross_entropy_1gpu.mat', {'cross_entropy': train_error})
		sio.savemat('/home/ruilan2/scratch/train_accuracy_1gpu.mat', {'accuracy': train_acc})

		data = deepGW.read_dataset(phase='test')

		test_snr = np.linspace(0.2, 3, 29)

		test_error = []
		test_acc = []


		for i in range(29):
			print("SNR = ", test_snr[i])

			input_epoch, label_epoch = deepGW.generate_batch_input(data, 'test', test_snr[i], test_step_size, 0, 0)

			input_batch, label_batch = deepGW.get_a_batch(input_epoch, label_epoch, test_step_size, 1, 0)

			# test = add_noise(testsig, testlabel, 100, test_snr[i], X, Y_)
			"""
			with tf.name_scope('Input'):
				X = tf.placeholder(tf.float32, [None, 1, 8192])

			with tf.name_scope('Label'):
				Y_ = tf.placeholder(tf.float32, [None, 1, 2])
			"""
			m, r = sess.run([loss, accuracy], feed_dict={X: input_batch, Y_: label_batch})
			print('test:' + ' cross_entropy:' + str(m) + ' accuracy:' + str(r))

			test_error.append(m)
			test_acc.append(r)

		sio.savemat('/home/ruilan2/scratch/test_cross_entropy_1gpu.mat', {'cross_entropy': test_error})
		sio.savemat('/home/ruilan2/scratch/test_accuracy_1gpu.mat', {'accuracy': test_acc})
	pass



inputs = deepGW.read_dataset(phase='train')
train(inputs)
