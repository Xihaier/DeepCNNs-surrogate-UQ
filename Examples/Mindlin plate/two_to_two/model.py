# Load modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from regressor import FR_25_beta


def FR_model(x_train, y_train, x_test, y_test, mp1, mp2, mp3, mp4, mp5, mp6, mp7, mp8):
    '''
    model parameters
    mp1: weight_decay
    mp2: batch_size
    mp3: learning_rate
    mp4: epochs
    mp5: eps
    '''
    weight_decay = mp1
    batch_size = mp2
    learning_rate = mp3
    epochs = mp4
    anneal_lr_freq = mp7
    anneal_lr_rate = mp8
    eps = mp5
    iters = int(np.floor(1024 / batch_size))
    kp = mp6
    test_freq = 10

    out1_train = np.zeros(epochs)
    out2_train = np.zeros(epochs)
    out3_train = np.zeros(epochs)
    out1_test = np.zeros(epochs)
    out2_test = np.zeros(epochs)
    out3_test = np.zeros(epochs)

    # initialize model
    input_x = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 2])
    output_y = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 2])
    is_training = tf.placeholder("bool", shape=[])
    lr = tf.placeholder(tf.float32, shape=[])
    keep_prob = tf.placeholder(tf.float32, shape=[])

    # model evaluation
    x = tf.transpose(tf.reshape(input_x, [-1, 2, 64, 64]), perm=[0, 2, 3, 1])
    y_hat = FR_25_beta(x, is_training, keep_prob)
    y = tf.transpose(tf.reshape(output_y, [-1, 2, 64, 64]), perm=[0, 2, 3, 1])

    # loss function 
    loss_l2 = tf.losses.mean_squared_error(y, y_hat)
    loss_weight = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss_weight = tf.multiply(loss_weight, weight_decay)
    loss_total = tf.add(loss_l2, loss_weight)

    # optimization
    infer_op = tf.train.AdamOptimizer(learning_rate = lr, epsilon = eps).minimize(loss_total)

    # prediction performance
    error = y_hat - y
    error_total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    error_unexplained = tf.reduce_sum(tf.square(tf.subtract(y, y_hat)))
    R2 = tf.maximum(1 - tf.div(error_unexplained, error_total), 0)
    
    # start training
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(1, epochs + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            
            p1s = []
            p2s = []
            p4s = []
            
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                train_feed_dict = {input_x: x_batch, output_y: y_batch, lr: learning_rate, is_training: True, keep_prob: kp}
                """
                output data
                p1: L2 loss
                p2: weight loss
                p3: difference between y and y_hat
                p4: coefficient of determination
                p5: y
                p6: y_hat
                """
                _, p1, p2, p3, p4, p5, p6 = sess.run([infer_op, loss_l2, loss_weight, error, R2, y, y_hat], feed_dict=train_feed_dict)
                p1s = np.append(p1s, p1)
                p2s = np.append(p2s, p2)
                p4s = np.append(p4s, p4)
            out1_train[epoch-1] = np.mean(p1s)
            out2_train[epoch-1] = np.mean(p2s)
            out3_train[epoch-1] = np.mean(p4s)
            print('Epoch {}: L2 loss = {}, Weight loss = {}, R2 = {}'.format(epoch, np.mean(p1s), np.mean(p2s), np.mean(p4s)))

            if epoch % test_freq == 0:
                """
                output data
                ppp1: L2 loss
                ppp2: weight loss
                ppp3: coefficient of determination
                """                
                test_feed_dict = {input_x: x_test, output_y: y_test, lr: learning_rate, is_training: False, keep_prob: kp}
                ppp1, ppp2, ppp3 = sess.run([loss_l2, loss_weight, R2], feed_dict=test_feed_dict)
                out1_test[epoch-1] = ppp1
                out2_test[epoch-1] = ppp2
                out3_test[epoch-1] = ppp3

            if epoch == epochs:
                test_feed_dict = {input_x: x_test, output_y: y_test, lr: learning_rate, is_training: False, keep_prob: kp}
                """
                output data
                pp1: difference between y and y_hat
                pp2: coefficient of determination
                pp3: y
                pp4: y_hat
                """
                pp1, pp2, pp3, pp4 = sess.run([error, R2, y, y_hat], feed_dict=test_feed_dict)
                sio.savemat('result.mat', {'error': pp1, 'R_2': pp2, 't_state': pp3, 'r_state': pp4, 'out1_train': out1_train, 'out2_train': out2_train, 'out3_train': out3_train, 'out1_test': out1_test, 'out2_test': out2_test, 'out3_test': out3_test})

