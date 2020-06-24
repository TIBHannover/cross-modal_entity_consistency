""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems

import numpy
import tensorflow
import cv2
import os


def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inumpyuts.
        if len(self.terminals) == 0:
            raise RuntimeError('No inumpyut variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_inumpyut = self.terminals[0]
        else:
            layer_inumpyut = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_inumpyut, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the inumpyut for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inumpyuts, trainable=True):
        # The inumpyut nodes for this network
        self.inumpyuts = inumpyuts
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inumpyuts)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        """Construct the network. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = numpy.load(data_path, encoding='latin1', allow_pickle=True).item()  # pylint: disable=no-member

        for op_name in data_dict:
            with tensorflow.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tensorflow.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        """Set the inumpyut(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tensorflow.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        """Verifies that the padding is one of the supported ones."""
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, inumpy, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding='SAME', group=1, biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the inumpyut
        c_i = int(inumpy.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given inumpyut and kernel
        convolve = lambda i, k: tensorflow.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tensorflow.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the inumpyut without any further complications.
            output = convolve(inumpy, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tensorflow.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tensorflow.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inumpy, name):
        with tensorflow.variable_scope(name):
            i = int(inumpy.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tensorflow.nn.relu(inumpy) + tensorflow.multiply(alpha, -tensorflow.nn.relu(-inumpy))
        return output

    @layer
    def max_pool(self, inumpy, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tensorflow.nn.max_pool(inumpy,
                                      ksize=[1, k_h, k_w, 1],
                                      strides=[1, s_h, s_w, 1],
                                      padding=padding,
                                      name=name)

    @layer
    def fc(self, inumpy, num_out, name, relu=True):
        with tensorflow.variable_scope(name):
            inumpyut_shape = inumpy.get_shape()
            if inumpyut_shape.ndims == 4:
                # The inumpyut is spatial. Vectorize it first.
                dim = 1
                for d in inumpyut_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tensorflow.reshape(inumpy, [-1, dim])
            else:
                feed_in, dim = (inumpy, inumpyut_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tensorflow.nn.relu_layer if relu else tensorflow.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """

    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tensorflow.reduce_max(target, axis, keepdims=True)
        target_exp = tensorflow.exp(target - max_axis)
        normalize = tensorflow.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tensorflow.div(target_exp, normalize, name)
        return softmax


class PNet(Network):

    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 10, 1, 1, padding='VALID', relu=False,
               name='conv1').prelu(name='PReLU1').max_pool(2, 2, 2, 2, name='pool1').conv(
                   3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2').prelu(name='PReLU2').conv(
                       3, 3, 32, 1, 1, padding='VALID', relu=False,
                       name='conv3').prelu(name='PReLU3').conv(1, 1, 2, 1, 1, relu=False,
                                                               name='conv4-1').softmax(3, name='prob1'))

        (self.feed('PReLU3')  # pylint: disable=no-value-for-parameter
         .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))


class RNet(Network):

    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 28, 1, 1, padding='VALID', relu=False,
               name='conv1').prelu(name='prelu1').max_pool(3, 3, 2, 2, name='pool1').conv(
                   3, 3, 48, 1, 1, padding='VALID', relu=False,
                   name='conv2').prelu(name='prelu2').max_pool(3, 3, 2, 2, padding='VALID', name='pool2').conv(
                       2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3').prelu(name='prelu3').fc(
                           128, relu=False, name='conv4').prelu(name='prelu4').fc(2, relu=False,
                                                                                  name='conv5-1').softmax(1,
                                                                                                          name='prob1'))

        (self.feed('prelu4')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv5-2'))


class ONet(Network):

    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False,
               name='conv1').prelu(name='prelu1').max_pool(3, 3, 2, 2, name='pool1').conv(
                   3, 3, 64, 1, 1, padding='VALID', relu=False,
                   name='conv2').prelu(name='prelu2').max_pool(3, 3, 2, 2, padding='VALID', name='pool2').conv(
                       3, 3, 64, 1, 1, padding='VALID', relu=False,
                       name='conv3').prelu(name='prelu3').max_pool(2, 2, 2, 2, name='pool3').conv(
                           2, 2, 128, 1, 1, padding='VALID', relu=False,
                           name='conv4').prelu(name='prelu4').fc(256, relu=False, name='conv5').prelu(name='prelu5').fc(
                               2, relu=False, name='conv6-1').softmax(1, name='prob1'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(10, relu=False, name='conv6-3'))


def create_mtcnn(sess, model_path):
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))

    with tensorflow.variable_scope('pnet'):
        data = tensorflow.placeholder(tensorflow.float32, (None, None, None, 3), 'inumpyut')
        pnet = PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tensorflow.variable_scope('rnet'):
        data = tensorflow.placeholder(tensorflow.float32, (None, 24, 24, 3), 'inumpyut')
        rnet = RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tensorflow.variable_scope('onet'):
        data = tensorflow.placeholder(tensorflow.float32, (None, 48, 48, 3), 'inumpyut')
        onet = ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)

    pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/inumpyut:0': img})
    rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/inumpyut:0': img})
    onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                    feed_dict={'onet/inumpyut:0': img})
    return pnet_fun, rnet_fun, onet_fun


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: inumpyut image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    max_boxes = 10000

    factor_count = 0
    total_boxes = numpy.empty((0, 9))
    points = numpy.empty(0)
    h = img.shape[0]
    w = img.shape[1]
    minl = numpy.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * numpy.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(numpy.ceil(h * scale))
        ws = int(numpy.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = numpy.expand_dims(im_data, 0)
        img_y = numpy.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = numpy.transpose(out[0], (0, 2, 1, 3))
        out1 = numpy.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = numpy.append(total_boxes, boxes, axis=0)

    if total_boxes.shape[0] > max_boxes:
        # TODO: Raise Exception
        return [], points

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = numpy.transpose(numpy.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = numpy.fix(total_boxes[:, 0:4]).astype(numpy.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = numpy.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = numpy.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return numpy.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = numpy.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out0 = numpy.transpose(out[0])
        out1 = numpy.transpose(out[1])
        score = out1[1, :]
        ipass = numpy.where(score > threshold[1])
        total_boxes = numpy.hstack([total_boxes[ipass[0], 0:4].copy(), numpy.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), numpy.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = numpy.fix(total_boxes).astype(numpy.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = numpy.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = numpy.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return numpy.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = numpy.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out0 = numpy.transpose(out[0])
        out1 = numpy.transpose(out[1])
        out2 = numpy.transpose(out[2])
        score = out2[1, :]
        points = out1
        ipass = numpy.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = numpy.hstack([total_boxes[ipass[0], 0:4].copy(), numpy.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = numpy.tile(w, (5, 1)) * points[0:5, :] + numpy.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = numpy.tile(h, (5, 1)) * points[5:10, :] + numpy.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), numpy.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points


def bulk_detect_face(images, detection_window_size_ratio, pnet, rnet, onet, threshold, factor):
    """Detects faces in a list of images
    images: list containing inumpyut images
    detection_window_size_ratio: ratio of minimum face size to smallest image dimension
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold [0-1]
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    all_scales = [None] * len(images)
    images_with_boxes = [None] * len(images)

    for i in range(len(images)):
        images_with_boxes[i] = {'total_boxes': numpy.empty((0, 9))}

    # create scale pyramid
    for index, img in enumerate(images):
        all_scales[index] = []
        h = img.shape[0]
        w = img.shape[1]
        minsize = int(detection_window_size_ratio * numpy.minimum(w, h))
        factor_count = 0
        minl = numpy.amin([h, w])
        if minsize <= 12:
            minsize = 12

        m = 12.0 / minsize
        minl = minl * m
        while minl >= 12:
            all_scales[index].append(m * numpy.power(factor, factor_count))
            minl = minl * factor
            factor_count += 1

    # # # # # # # # # # # # #
    # first stage - fast proposal network (pnet) to obtain face candidates
    # # # # # # # # # # # # #

    images_obj_per_resolution = {}

    # TODO: use some type of rounding to number module 8 to increase probability that pyramid images will have the
    #       same resolution across inumpyut images

    for index, scales in enumerate(all_scales):
        h = images[index].shape[0]
        w = images[index].shape[1]

        for scale in scales:
            hs = int(numpy.ceil(h * scale))
            ws = int(numpy.ceil(w * scale))

            if (ws, hs) not in images_obj_per_resolution:
                images_obj_per_resolution[(ws, hs)] = []

            im_data = imresample(images[index], (hs, ws))
            im_data = (im_data - 127.5) * 0.0078125
            img_y = numpy.transpose(im_data, (1, 0, 2))  # caffe uses different dimensions ordering
            images_obj_per_resolution[(ws, hs)].append({'scale': scale, 'image': img_y, 'index': index})

    for resolution in images_obj_per_resolution:
        images_per_resolution = [i['image'] for i in images_obj_per_resolution[resolution]]
        outs = pnet(images_per_resolution)

        for index in range(len(outs[0])):
            scale = images_obj_per_resolution[resolution][index]['scale']
            image_index = images_obj_per_resolution[resolution][index]['index']
            out0 = numpy.transpose(outs[0][index], (1, 0, 2))
            out1 = numpy.transpose(outs[1][index], (1, 0, 2))

            boxes, _ = generateBoundingBox(out1[:, :, 1].copy(), out0[:, :, :].copy(), scale, threshold[0])

            # inter-scale nms
            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                images_with_boxes[image_index]['total_boxes'] = numpy.append(
                    images_with_boxes[image_index]['total_boxes'], boxes, axis=0)

    for index, image_obj in enumerate(images_with_boxes):
        numbox = image_obj['total_boxes'].shape[0]
        if numbox > 0:
            h = images[index].shape[0]
            w = images[index].shape[1]
            pick = nms(image_obj['total_boxes'].copy(), 0.7, 'Union')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            regw = image_obj['total_boxes'][:, 2] - image_obj['total_boxes'][:, 0]
            regh = image_obj['total_boxes'][:, 3] - image_obj['total_boxes'][:, 1]
            qq1 = image_obj['total_boxes'][:, 0] + image_obj['total_boxes'][:, 5] * regw
            qq2 = image_obj['total_boxes'][:, 1] + image_obj['total_boxes'][:, 6] * regh
            qq3 = image_obj['total_boxes'][:, 2] + image_obj['total_boxes'][:, 7] * regw
            qq4 = image_obj['total_boxes'][:, 3] + image_obj['total_boxes'][:, 8] * regh
            image_obj['total_boxes'] = numpy.transpose(
                numpy.vstack([qq1, qq2, qq3, qq4, image_obj['total_boxes'][:, 4]]))
            image_obj['total_boxes'] = rerec(image_obj['total_boxes'].copy())
            image_obj['total_boxes'][:, 0:4] = numpy.fix(image_obj['total_boxes'][:, 0:4]).astype(numpy.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(image_obj['total_boxes'].copy(), w, h)

            numbox = image_obj['total_boxes'].shape[0]
            tempimg = numpy.zeros((24, 24, 3, numbox))

            if numbox > 0:
                for k in range(0, numbox):
                    tmp = numpy.zeros((int(tmph[k]), int(tmpw[k]), 3))
                    tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = images[index][y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                        tempimg[:, :, :, k] = imresample(tmp, (24, 24))
                    else:
                        return numpy.empty()

                tempimg = (tempimg - 127.5) * 0.0078125
                image_obj['rnet_inumpyut'] = numpy.transpose(tempimg, (3, 1, 0, 2))

    # # # # # # # # # # # # #
    # second stage - refinement of face candidates with rnet
    # # # # # # # # # # # # #

    bulk_rnet_inumpyut = numpy.empty((0, 24, 24, 3))
    for index, image_obj in enumerate(images_with_boxes):
        if 'rnet_inumpyut' in image_obj:
            bulk_rnet_inumpyut = numpy.append(bulk_rnet_inumpyut, image_obj['rnet_inumpyut'], axis=0)

    out = rnet(bulk_rnet_inumpyut)
    out0 = numpy.transpose(out[0])
    out1 = numpy.transpose(out[1])
    score = out1[1, :]

    i = 0
    for index, image_obj in enumerate(images_with_boxes):
        if 'rnet_inumpyut' not in image_obj:
            continue

        rnet_inumpyut_count = image_obj['rnet_inumpyut'].shape[0]
        score_per_image = score[i:i + rnet_inumpyut_count]
        out0_per_image = out0[:, i:i + rnet_inumpyut_count]

        ipass = numpy.where(score_per_image > threshold[1])
        image_obj['total_boxes'] = numpy.hstack(
            [image_obj['total_boxes'][ipass[0], 0:4].copy(),
             numpy.expand_dims(score_per_image[ipass].copy(), 1)])

        mv = out0_per_image[:, ipass[0]]

        if image_obj['total_boxes'].shape[0] > 0:
            h = images[index].shape[0]
            w = images[index].shape[1]
            pick = nms(image_obj['total_boxes'], 0.7, 'Union')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            image_obj['total_boxes'] = bbreg(image_obj['total_boxes'].copy(), numpy.transpose(mv[:, pick]))
            image_obj['total_boxes'] = rerec(image_obj['total_boxes'].copy())

            numbox = image_obj['total_boxes'].shape[0]

            if numbox > 0:
                tempimg = numpy.zeros((48, 48, 3, numbox))
                image_obj['total_boxes'] = numpy.fix(image_obj['total_boxes']).astype(numpy.int32)
                dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(image_obj['total_boxes'].copy(), w, h)

                for k in range(0, numbox):
                    tmp = numpy.zeros((int(tmph[k]), int(tmpw[k]), 3))
                    tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = images[index][y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                        tempimg[:, :, :, k] = imresample(tmp, (48, 48))
                    else:
                        return numpy.empty()
                tempimg = (tempimg - 127.5) * 0.0078125
                image_obj['onet_inumpyut'] = numpy.transpose(tempimg, (3, 1, 0, 2))

        i += rnet_inumpyut_count

    # # # # # # # # # # # # #
    # third stage - further refinement and facial landmarks positions with onet
    # # # # # # # # # # # # #

    bulk_onet_inumpyut = numpy.empty((0, 48, 48, 3))
    for index, image_obj in enumerate(images_with_boxes):
        if 'onet_inumpyut' in image_obj:
            bulk_onet_inumpyut = numpy.append(bulk_onet_inumpyut, image_obj['onet_inumpyut'], axis=0)

    out = onet(bulk_onet_inumpyut)

    out0 = numpy.transpose(out[0])
    out1 = numpy.transpose(out[1])
    out2 = numpy.transpose(out[2])
    score = out2[1, :]
    points = out1

    i = 0
    ret = []
    for index, image_obj in enumerate(images_with_boxes):
        if 'onet_inumpyut' not in image_obj:
            ret.append(None)
            continue

        onet_inumpyut_count = image_obj['onet_inumpyut'].shape[0]

        out0_per_image = out0[:, i:i + onet_inumpyut_count]
        score_per_image = score[i:i + onet_inumpyut_count]
        points_per_image = points[:, i:i + onet_inumpyut_count]

        ipass = numpy.where(score_per_image > threshold[2])
        points_per_image = points_per_image[:, ipass[0]]

        image_obj['total_boxes'] = numpy.hstack(
            [image_obj['total_boxes'][ipass[0], 0:4].copy(),
             numpy.expand_dims(score_per_image[ipass].copy(), 1)])
        mv = out0_per_image[:, ipass[0]]

        w = image_obj['total_boxes'][:, 2] - image_obj['total_boxes'][:, 0] + 1
        h = image_obj['total_boxes'][:, 3] - image_obj['total_boxes'][:, 1] + 1
        points_per_image[0:5, :] = numpy.tile(w, (5, 1)) * points_per_image[0:5, :] + numpy.tile(
            image_obj['total_boxes'][:, 0], (5, 1)) - 1
        points_per_image[5:10, :] = numpy.tile(h, (5, 1)) * points_per_image[5:10, :] + numpy.tile(
            image_obj['total_boxes'][:, 1], (5, 1)) - 1

        if image_obj['total_boxes'].shape[0] > 0:
            image_obj['total_boxes'] = bbreg(image_obj['total_boxes'].copy(), numpy.transpose(mv))
            pick = nms(image_obj['total_boxes'].copy(), 0.7, 'Min')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            points_per_image = points_per_image[:, pick]

            ret.append((image_obj['total_boxes'], points_per_image))
        else:
            ret.append(None)

        i += onet_inumpyut_count

    return ret


# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox, reg):
    """Calibrate bounding boxes"""
    if reg.shape[1] == 1:
        reg = numpy.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = numpy.transpose(numpy.vstack([b1, b2, b3, b4]))
    return boundingbox


def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride = 2
    cellsize = 12

    imap = numpy.transpose(imap)
    dx1 = numpy.transpose(reg[:, :, 0])
    dy1 = numpy.transpose(reg[:, :, 1])
    dx2 = numpy.transpose(reg[:, :, 2])
    dy2 = numpy.transpose(reg[:, :, 3])
    y, x = numpy.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = numpy.flipud(dx1)
        dy1 = numpy.flipud(dy1)
        dx2 = numpy.flipud(dx2)
        dy2 = numpy.flipud(dy2)
    score = imap[(y, x)]
    reg = numpy.transpose(numpy.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = numpy.empty((0, 3))
    bb = numpy.transpose(numpy.vstack([y, x]))
    q1 = numpy.fix((stride * bb + 1) / scale)
    q2 = numpy.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = numpy.hstack([q1, q2, numpy.expand_dims(score, 1), reg])
    return boundingbox, reg


# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size == 0:
        return numpy.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = numpy.argsort(s)
    pick = numpy.zeros_like(s, dtype=numpy.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = numpy.maximum(x1[i], x1[idx])
        yy1 = numpy.maximum(y1[i], y1[idx])
        xx2 = numpy.minimum(x2[i], x2[idx])
        yy2 = numpy.minimum(y2[i], y2[idx])
        w = numpy.maximum(0.0, xx2 - xx1 + 1)
        h = numpy.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / numpy.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[numpy.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(numpy.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(numpy.int32)
    numbox = total_boxes.shape[0]

    dx = numpy.ones((numbox), dtype=numpy.int32)
    dy = numpy.ones((numbox), dtype=numpy.int32)
    edx = tmpw.copy().astype(numpy.int32)
    edy = tmph.copy().astype(numpy.int32)

    x = total_boxes[:, 0].copy().astype(numpy.int32)
    y = total_boxes[:, 1].copy().astype(numpy.int32)
    ex = total_boxes[:, 2].copy().astype(numpy.int32)
    ey = total_boxes[:, 3].copy().astype(numpy.int32)

    tmp = numpy.where(ex > w)
    edx.flat[tmp] = numpy.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = numpy.where(ey > h)
    edy.flat[tmp] = numpy.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = numpy.where(x < 1)
    dx.flat[tmp] = numpy.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = numpy.where(y < 1)
    dy.flat[tmp] = numpy.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = numpy.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + numpy.transpose(numpy.tile(l, (2, 1)))
    return bboxA


def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)  # @UndefinedVariable
    return im_data

    # This method is kept for debugging purpose


#     h=img.shape[0]
#     w=img.shape[1]
#     hs, ws = sz
#     dx = float(w) / ws
#     dy = float(h) / hs
#     im_data = numpy.zeros((hs,ws,3))
#     for a1 in range(0,hs):
#         for a2 in range(0,ws):
#             for a3 in range(0,3):
#                 im_data[a1,a2,a3] = img[int(floor(a1*dy)),int(floor(a2*dx)),a3]
#     return im_data
