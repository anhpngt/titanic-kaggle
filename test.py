import os
import tensorflow as tf
import numpy as np
import csv

import main

save_dir = 'saved_model'
dataset_dir = 'datasets' # Folder containing data sets
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

if __name__=='__main__':
    # Get feature average to substitute for missing samples
    feature_averages =  main.getAverages(os.path.join(dataset_dir, 'train.csv'), ['Age'])
    
    # Import testing data
    data = main.getFeatures(os.path.join(dataset_dir, 'test.csv'), features, feature_averages)
    teX = np.array(data)
    
    # Create the model
    feature_size = len(features)
    x = tf.placeholder(tf.float32, [None, feature_size])
    W = tf.Variable(tf.zeros([feature_size, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    
    # Start the session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Import saved model
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(save_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
#     print(sess.run(tf.all_variables()))
#     prediction = sess.run([y], feed_dict={x: teX})
    