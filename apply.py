#!/usr/bin/env python


import numpy as np
import tensorflow as tf
import cv2
import argparse
import os


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Apply network on an input image.')
    parser.add_argument('input_image', type=str,
            help='sum the integers (default: find the max)')
    args = parser.parse_args()

    # Setup Tensorflow session with previously trained graph
    config = tf.ConfigProto()
    sess = tf.Session()

    # NOTE: In this path should be a file `checkpoint`, which links to the latest model files
    path_weights = tf.train.latest_checkpoint('renamed/')
    path_graph = '{}.meta'.format(path_weights)

    print('Restore graph from file: {}'.format(path_graph))
    saver = tf.train.import_meta_graph(path_graph)

    print('Restore weights from file: {}'.format(path_weights))
    saver.restore(sess, path_weights)

    # Get input and output nodes
    image = tf.get_default_graph().get_tensor_by_name('input_image:0')
    output = tf.get_default_graph().get_tensor_by_name('output_softmax:0')

    # Feed an image and test the result
    if not os.path.exists(args.input_image):
        raise Exception('Input does not exist: {}'.format(args.input_image))

    x = cv2.imread(args.input_image)
    y_= sess.run(output, feed_dict={image:x})
    y_ = y_.reshape((x.shape[0], x.shape[1], 2))
    y = np.zeros(x.shape, dtype=np.uint8)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            prob = y_[i,j,0]
            # Road: 1
            if prob > 0.5:
                y[i,j] = (int(255*prob), 0, 0)
            # Everything else: 0
            else:
                y[i,j] = (0, 0, int(255*(1.0-prob)))

    z = np.zeros(x.shape, dtype=np.uint8)
    cv2.addWeighted(y, 0.5, x, 0.5, 0.0, z)
    cv2.imwrite('apply_{}'.format(os.path.basename(args.input_image)),
            np.vstack((x,y,z)))
