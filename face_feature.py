'''
@Author: David Vu
Run the pretrained model to extract 128D face features
'''

import tensorflow as tf
from tensorflow.python.platform import gfile
from architecture import inception_resnet_v1 as resnet
import numpy as np

scope_variable = 'onet/conv1/weights:0'

model_pb = True

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph



class FaceFeature(object):
    def __init__(self, face_rec_graph, save_part, model_path = 'models/model-20170512-110547.ckpt-250000'):
        '''

        :param face_rec_sess: FaceRecSession object
        :param model_path:
        '''
        if save_part == True:
            print("Loading model...")
            with face_rec_graph.graph.as_default():
                self.sess = tf.Session() #face_rec_graph.sess #
                self.sess.run(tf.global_variables_initializer())

                self.x = tf.placeholder('float', [None,160,160,3],name='input'); #default input for the NN is 160x160x3
                self.embeddings = tf.nn.l2_normalize(
                                            resnet.inference(self.x, 0.6, phase_train=False)[0], 1, 1e-10); #some magic numbers that u dont have to care about
                print("self.embeddings")
                print(self.embeddings)
                saver = tf.train.Saver() #saver load pretrain model
                saver.restore(self.sess, model_path)
                print("Model loaded")

                writer = tf.summary.FileWriter("/tmp/model4/FaceFeature/", self.sess.graph)
                saver = tf.train.Saver() #saver load pretrain model
                save_path = tf.train.Saver().save(self.sess, "/tmp/model4/FaceFeature/model.ckpt")
                print("Model saved in path: %s" % save_path)
                self.features = lambda img: self.sess.run(('l2_normalize:0'), feed_dict = {"input:0" : img})

        else:
                if model_pb == True :
                    print("Load FaceFeature from saved pb file ")

                    model_filename ='/tmp/model4/FaceFeature/graph/graph_FaceFeature.pb'
                    g_in =  load_graph(model_filename)
                    self.sess  =  tf.Session(graph= g_in)
                    # self.sess.run(tf.global_variables_initializer())

                    print("///////////////////llllll")
                    for v in tf.trainable_variables():
                        # if v.name == scope_variable:
                            print(v)
                            print("/******************/")
                            print(self.sess.run(v))


                    for v in self.sess.graph.get_operations():
                        print(v)
                        print("/******************/")
                            # print(self.sess.run(v))
                else:
                    print("Load FaceFeature from saved model ")
                    self.sess = tf.Session()
                    self.sess.run(tf.global_variables_initializer())
                    model_path = "/tmp/model4/FaceFeature/model.ckpt"
                    model_path_p = "/tmp/model4/FaceFeature/"
                    new_saver = tf.train.import_meta_graph(model_path + '.meta')
                    new_saver.restore(self.sess,tf.train.latest_checkpoint(model_path_p))

                self.features = lambda img: self.sess.run(('prefix/l2_normalize:0'), feed_dict = {"prefix/input:0" : img})





    def get_features(self, input_imgs):
        # print("input_imgs.shape")
        # print(input_imgs)
        images = load_data_list(input_imgs,160)

        images = images.astype(float)

        # return self.sess.run(self.embeddings, feed_dict = {self.x : images})
        # return self.sess.run(('l2_normalize:0'), feed_dict = {"Placeholder:0" : images})
        return self.features(images)



#some image preprocess stuff
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def load_data_list(imgList, image_size, do_prewhiten=True):
    images = np.zeros((len(imgList), image_size, image_size, 3))
    i = 0
    for img in imgList:
        if img is not None:
            if do_prewhiten:
                img = prewhiten(img)
            images[i, :, :, :] = img
            i += 1
    return images
