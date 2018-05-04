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
        # else:
            # with  tf.Session() as self.sess:
                # new_saver = tf.train.import_meta_graph(model_path + '.meta')
                # # new_saver.restore(self.sess, model_path)
                # new_saver.restore(self.sess,tf.train.latest_checkpoint(model_path_p))

                # print("Model restored :")
                # self.graph = self.sess.graph  #tf.get_default_graph();
                # # self.graph.as_default()
                # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                # for v in tf.trainable_variables():
                #     if v.name == scope_variable:
                #         # print(v)
                #         print("/******************/")
                #         print(self.sess.run(v))


                # for op in tf.get_default_graph().get_operations():
                #     print(str(op.name))




