'''
Load pretrain models and create a tensorflow session to run them

@Author: David Vu
'''
import tensorflow as tf


model_path = "/tmp/model4/model.ckpt"
model_path_p = "/tmp/model4/"
scope_variable = 'onet/conv1/weights:0'

class FaceRecGraph(object):
    def __init__(self, save_part):
        '''
            There'll be more to come in this class
        '''

        if save_part == True:
            self.graph = tf.Graph();





