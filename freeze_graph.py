import numpy as np
import tensorflow as tf
import os



print("Generate graph_MTCNN.pb ...")
with  tf.Session() as sess:
    model_path = "/tmp/model4/MTCNN/model.ckpt"
    model_path_p = "/tmp/model4/MTCNN/"
    sess.run(tf.global_variables_initializer())
    var = model_path + '.meta'
    new_saver = tf.train.import_meta_graph(model_path + '.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint(model_path_p))

    output_node_names='pnet/conv4-2/BiasAdd,pnet/prob1,pnet/input,rnet/conv5-2/conv5-2,rnet/prob1,rnet/input,onet/conv6-2/conv6-2,onet/conv6-3/conv6-3,onet/prob1,onet/input'


    output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session
                sess.graph.as_graph_def(), # input_graph_def is useful for retrieving the nodes
                output_node_names.split(",")  )

    output_graph="/tmp/model4/MTCNN/graph/"
    if not os.path.exists(output_graph):
        os.makedirs(output_graph)
    with tf.gfile.GFile(os.path.join(output_graph ,'graph_MTCNN.pb'), "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()


print("Generate graph_FaceFeature.pb ...")
with  tf.Session() as sess:
    model_path = "/tmp/model4/FaceFeature/model.ckpt"
    model_path_p = "/tmp/model4/FaceFeature/"
    sess.run(tf.global_variables_initializer())
    var = model_path + '.meta'
    new_saver = tf.train.import_meta_graph(model_path + '.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint(model_path_p))

    output_node_names='input,l2_normalize'

    output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session
                sess.graph.as_graph_def(), # input_graph_def is useful for retrieving the nodes
                output_node_names.split(",")  )

    output_graph="/tmp/model4/FaceFeature/graph/"
    if not os.path.exists(output_graph):
        os.makedirs(output_graph)
    with tf.gfile.GFile(os.path.join(output_graph ,'graph_FaceFeature.pb'), "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()