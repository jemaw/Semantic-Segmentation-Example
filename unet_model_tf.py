from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope


def unet_inference(images,
                kernel_num=32,
                is_training=True,
                reuse=None,
                num_output_channels=1,
                style='default',
                normalizer_fn=None,
                normalizer_params=None,
                double_conv=False,
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
                weights_initializer=tf.contrib.layers.xavier_initializer()):

    normalizer_params['is_training'] = is_training


    with arg_scope([tf.contrib.layers.conv2d],
                   normalizer_fn=normalizer_fn,
                   normalizer_params=normalizer_params,
                   kernel_size=[3,3],
                   weights_initializer=weights_initializer,
                   weights_regularizer=weights_regularizer,
                   trainable=is_training):
        with arg_scope([tf.contrib.layers.conv2d_transpose],
                   normalizer_fn=normalizer_fn,
                   normalizer_params=normalizer_params,
                   kernel_size=[3,3],
                   stride=2,
                   weights_initializer=weights_initializer,
                   weights_regularizer=weights_regularizer,
                   trainable=is_training):

            with tf.variable_scope('unet', reuse=reuse):
                return unet(images,
                           kernel_num=kernel_num,
                           reuse=reuse,
                           double_conv=double_conv,
                           num_output_channels=num_output_channels)


def unet(images,
         kernel_num,
         reuse,
         num_output_channels,
         double_conv=False):

    end_points = {}

    # first
    with tf.variable_scope('in1', reuse=reuse):
        in1 = tf.contrib.layers.conv2d(images, kernel_num)
        if double_conv:
            in1 = tf.contrib.layers.conv2d(in1, kernel_num)

    with tf.variable_scope('in2', reuse=reuse):
        # 2 * kernel_num
        in2 = tf.layers.max_pooling2d(in1, [2,2], strides=2)
        in2 = tf.contrib.layers.conv2d(in2, 2*kernel_num)
        if double_conv:
            in2 = tf.contrib.layers.conv2d(in2, 2*kernel_num)

    with tf.variable_scope('in4', reuse=reuse):
        # 4 * kernel_num
        in4 = tf.layers.max_pooling2d(in2, [2,2], strides=2)
        in4 = tf.contrib.layers.conv2d(in4, 4*kernel_num)
        if double_conv:
            in4 = tf.contrib.layers.conv2d(in4, 4*kernel_num)

    with tf.variable_scope('in8', reuse=reuse):
        # 8 * kernel_num
        in8 = tf.layers.max_pooling2d(in4, [2,2], strides=2)
        in8 = tf.contrib.layers.conv2d(in8, 8*kernel_num)
        if double_conv:
            in8 = tf.contrib.layers.conv2d(in8, 8*kernel_num)

    with tf.variable_scope('in16', reuse=reuse):
        # 16 * kernel_num
        in16 = tf.layers.max_pooling2d(in8, [2,2], strides=2)
        in16 = tf.contrib.layers.conv2d(in16, 16*kernel_num)
        if double_conv:
            in16 = tf.contrib.layers.conv2d(in16, 16*kernel_num)

    with tf.variable_scope('in_n_out', reuse=reuse):
        last_in = in16
        # in and out
        in_out_kernels = 32*kernel_num
        in_n_out = tf.layers.max_pooling2d(last_in, [2,2], strides=2)
        in_n_out = tf.contrib.layers.conv2d(in_n_out, in_out_kernels)
        in_n_out = tf.contrib.layers.conv2d(in_n_out, in_out_kernels)
        in_n_out = tf.contrib.layers.conv2d_transpose(in_n_out, in_out_kernels)

        # concat
        in_n_out = tf.concat([in_n_out, last_in ], 3)

    with tf.variable_scope('out16', reuse=reuse):
        # out 16 * kernel_num 
        out16 = tf.contrib.layers.conv2d(in_n_out, 16*kernel_num)
        if double_conv:
            out16 = tf.contrib.layers.conv2d(out16, 16*kernel_num)
        out16 = tf.contrib.layers.conv2d_transpose(out16, 16*kernel_num)
        out16 = tf.concat([out16, in8], 3)

    with tf.variable_scope('out8', reuse=reuse):
        # out 8 * kernel_num 
        out8 = tf.contrib.layers.conv2d(out16, 8*kernel_num)
        if double_conv:
            out8 = tf.contrib.layers.conv2d(out8, 8*kernel_num)
        out8 = tf.contrib.layers.conv2d_transpose(out8, 8*kernel_num)
        out8 = tf.concat([out8, in4], 3)

    with tf.variable_scope('out4', reuse=reuse):
        # out 4 * kernel_num 
        out4 = tf.contrib.layers.conv2d(out8, 4*kernel_num)
        if double_conv:
            out16 = tf.contrib.layers.conv2d(out4, 4*kernel_num)
        out4 = tf.contrib.layers.conv2d_transpose(out4, 4*kernel_num)
        out4 = tf.concat([out4, in2], 3)

    with tf.variable_scope('out2', reuse=reuse):
        # out 2 * kernel_num 
        out2 = tf.contrib.layers.conv2d(out4, 2*kernel_num)
        if double_conv:
            out16 = tf.contrib.layers.conv2d(out2, 2*kernel_num)
        out2 = tf.contrib.layers.conv2d_transpose(out2, 2*kernel_num)
        out2 = tf.concat([out2, in1], 3)

    with tf.variable_scope('out1', reuse=reuse):
        # out 1 * kernel_num 
        out1 = tf.contrib.layers.conv2d(out2, kernel_num)
        if double_conv:
            out1 = tf.contrib.layers.conv2d(out1, kernel_num)

    # 1x1 to get to num_output_channels
    with tf.variable_scope('1x1', reuse=reuse):
        net = tf.contrib.layers.conv2d(out1,
                                       num_output_channels,
                                       kernel_size=[1,1],
                                       activation_fn=None,
                                       normalizer_fn=None)
    logits = net
    prediction = tf.nn.sigmoid(logits)
    end_points['logits'] = logits
    end_points['prediction'] = prediction


    return {'logits': logits, 'probs': prediction}, end_points

