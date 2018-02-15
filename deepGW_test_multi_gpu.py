import tensorflow as tf
import deepGW
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


test_step_size = 128
all_num_gpus = [1]    # testing      ########################
log_device_placement = False
num_gpus = 1

def make_plot(loss, acc):
    """
    Test step: plot SNR vs. MSE, SNR vs. realative_error
    :param loss(list): list of MSEs
    :param acc(list): list of relative error
    :return: None
    """
    snr = np.linspace(0.2, 3, 29)
    for i in range(len(all_num_gpus)):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(snr, loss[i])
        plt.ylabel("MSE")
        ax2 = fig.add_subplot(212)
        ax2.plot(snr, acc[i])
        plt.xlabel("SNR")
        plt.ylabel("Relative Error")
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.suptitle("Test Prediction on " + str(all_num_gpus[i]) + " GPUs")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("result_img/Test_" + str(all_num_gpus[i]) + "_GPUs")


def test(inputs, num_gpus):
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, [None, 1, 8192])
    with tf.name_scope('Label'):
        Y_ = tf.placeholder(tf.float32, [None, 1, 2])

    inputs_tensor = deepGW.convert_to_tensor_float(X[:, 0, :])
    labels_tensor = deepGW.convert_to_tensor_float(Y_[:, 0, :])

    pred, logits = deepGW.inference(inputs_tensor)    # same model for inference

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
    # cross_entropy = tf.reduce_mean(cross_entropy)
    # is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) * 100

    loss = deepGW.loss(logits, labels_tensor)
    acc = deepGW.accuracy(pred, labels_tensor)


    sensitivity, opt = tf.metrics.sensitivity_at_specificity(predictions=pred, labels=labels_tensor, specificity=0.99)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver()
    if num_gpus == 1:
        model_path = "/home/ruilan2/multi-gpu/wave_final/upload_results/save_proj_2gpus.ckpt-3999"
    else:
        model_path = "/home/ruilan2/multi-gpu/wave_large/save_results2/save_proj_4gpus.ckpt-7999"
    #model_path = "/home/abc99lr/Gravatitional-Wave-Prediction/mass_small/Model/Model_" + str(num_gpus) + "_GPU.ckpt"
    saver.restore(sess, model_path)

    test_snr = np.linspace(0.2, 3, 29)
    test_loss = []
    test_acc = []
    test_sns = []

    for i in range(29):
        snr = test_snr[i]
        # print("SNR = ", snr)
        input_epoch, label_epoch = deepGW.generate_batch_input(inputs, 'test', snr,
                                                                         1*test_step_size, 0, 0)
        input_batch, label_batch = deepGW.get_a_batch(input_epoch, label_epoch, test_step_size, 1, 0)

        # test = add_noise(testsig, testlabel, 100, test_snr[i], X, Y_)

        cur_loss, cur_acc = sess.run([loss, acc], feed_dict={X: input_batch, Y_: label_batch})
        #sns = sess.run([opt], feed_dict={X: input_batch, Y_: label_batch})

        print('test: SNR =' + str(snr) + ' cross_entropy:' + str(cur_loss) + ' accuracy:' + str(cur_acc))
        #print('sensitivity=' + str(sns))

        test_loss.append(cur_loss)
        test_acc.append(cur_acc)
        #test_sns.append(sns)

    fig = plt.figure()
    plt.plot(test_snr, np.array(test_acc).tolist())
    plt.xlabel('Signal-to-Noise Ratio')
    plt.ylabel('Accuracy of Classifier (%)')
    #plt.title('SNR - Accuracy Plot')
    plt.savefig("./result_img/snr-acc.png")


    return test_loss, test_acc

    #sio.savemat('/home/ruilan2/multi-gpu/wave_large/save_results/cross_entropy.mat', {'cross_entropy': test_error})
    #sio.savemat('/home/ruilan2/multi-gpu/wave_large/save_results/accuracy.mat', {'accuracy': test_acc})



if __name__ == "__main__":
    inputs = deepGW.read_dataset(phase='test')
    loss1, acc1 = [], []
    #for i in range(len(all_num_gpus)):
    ret_value = test(inputs, 1)
    loss1.append(ret_value[0])
    acc1.append(ret_value[1])

    """
    loss2, acc2 = [], []
    ret_value = test(inputs, 2)
    loss2.append(ret_value[0])
    acc2.append(ret_value[1])
    #make_plot(loss, acc)

    test_snr = np.linspace(0.2, 3, 29)
    fig, ax = plt.subplots()
    plt.xticks(np.linspace(0.2, 3, 8))
    ax.plot(test_snr, np.array(acc1).T.tolist(), 'b', label="Using 1 GPU")
    ax.plot(test_snr, np.array(acc2).T.tolist(), 'r', label="Using 2 GPUs")
    legend = ax.legend(loc='lower right', shadow=False)
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.title('SNR - Accuracy Plot')
    plt.savefig("result_img/acc")

    fig, ax = plt.subplots()
    plt.xticks(np.linspace(0.2, 3, 8))
    ax.plot(test_snr, np.array(loss1).T.tolist(), 'b', label="Using 1 GPU")
    ax.plot(test_snr, np.array(loss2).T.tolist(), 'r', label="Using 2 GPUs")
    legend = ax.legend(loc='upper right', shadow=False)
    plt.xlabel('SNR')
    plt.ylabel('Cross-entropy')
    plt.title('SNR - Cross-entropy Plot')
    plt.savefig("result_img/loss")
    """