import tensorflow as tf

from networks import conv2d
from networks import dense_block
from networks import upsampling

'''field regressor for two-to-two mapping
data flow mechanism: 64*64*2 ---> 31*31*64 ---> 31*31*144
                             ---> 17*17*64 ---> 17*17*144 
                             ---> 32*32*72 ---> 32*32*144
                             ---> 64*64*72 ---> 64*64*144 ---> 64*64*2 
'''
def FR_25_beta(x, is_training, keep_prob):
    # layer 1
    pad1 = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    y = conv2d(x, 2, 64, 7, [1, 2, 2, 1], pad1, "VALID")

    # layer 2~6
    pad2 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    y, _ = dense_block(y, 5, 64, 16, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob)
 
    # layer 7
    y = conv2d(y, 144, 64, 3, [1, 2, 2, 1], pad2, "VALID")

    # layer 8~12
    y, _ = dense_block(y, 5, 64, 16, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob)

    # layer 13~14
    y = upsampling(y, 144, 72, 3, [1, 1, 1, 1], pad2, "VALID", 32, 32, is_training)

    # layer 15~18
    y, _ = dense_block(y, 4, 72, 18, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob)

    # layer 19~20
    y = upsampling(y, 144, 72, 3, [1, 1, 1, 1], pad2, "VALID", 64, 64, is_training)

    # layer 21~24
    y, _ = dense_block(y, 4, 72, 18, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob)
    
    # layer 25
    y = conv2d(y, 144, 2, 3, [1, 1, 1, 1], pad2, "VALID")
    return y





