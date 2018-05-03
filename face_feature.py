'''
@Author: David Vu
Run the pretrained model to extract 128D face features
'''

import tensorflow as tf
from architecture import inception_resnet_v1 as resnet
import numpy as np

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

                self.x = tf.placeholder('float', [None,160,160,3]); #default input for the NN is 160x160x3
                self.embeddings = tf.nn.l2_normalize(
                                            resnet.inference(self.x, 0.6, phase_train=False)[0], 1, 1e-10); #some magic numbers that u dont have to care about
                print("self.embeddings")
                print(self.embeddings)
                saver = tf.train.Saver() #saver load pretrain model
                saver.restore(self.sess, model_path)
                print("Model loaded")
        else:
            with face_rec_graph.graph.as_default():
                self.sess = tf.Session() #face_rec_graph.sess #
                self.sess.run(tf.global_variables_initializer())
                self.x = tf.placeholder('float', [None,160,160,3]); #default input for the NN is 160x160x3



    def get_features(self, input_imgs):
        print("input_imgs.shape")
        print(input_imgs)
        images = load_data_list(input_imgs,160)

        print("images.shape")
        print(images.shape)
        # return self.sess.run(self.embeddings, feed_dict = {self.x : images})
        return self.sess.run(('l2_normalize:0'), feed_dict = {self.x : images})



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
