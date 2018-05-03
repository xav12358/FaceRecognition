import tensorflow as tf

model_path = "/tmp/model4/model.ckpt"
x = tf.placeholder('float', [None,160,160,3]); #default input for the NN is 160x160x3
# saver = tf.train.Saver() #saver load pretrain model
with tf.Session()  as sess:#face_rec_graph.sess #
    # sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('/tmp/model4/model.ckpt.meta')
    new_saver.restore(sess, model_path)
    print("Model loaded")

    # for op in tf.get_default_graph().get_operations():
    #     print(str(op.name))
