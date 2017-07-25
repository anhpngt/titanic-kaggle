import os
import csv
import tensorflow as tf
import numpy as np


def getAverages(test_file_path, features):
    avrg = []
    with open(test_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for feature in features:
            count = 0
            total = 0
            for row in reader:
                try:
                    total += float(row[feature])
                except ValueError:
                    continue
                count += 1
            if count == 0:
                avrg.append({feature: 0})
                continue
            avrg.append({feature: round(total/count, 1)})
    return avrg[0]
    
def getFeatures(test_file_path, features, feature_averages):
    x = []
    with open(test_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            new_data = []
            for feature in features:
                    if feature == 'Sex':
                        new_data.append(0 if row['Sex'] == 'female' else 1)
                    elif feature == 'Embarked':
                        new_data.append(0 if row['Embarked'] == 'C' else 1 if row['Embarked'] == 'Q' else 2)
                    else:
                        try:
                            temp = float(row[feature])
                            new_data.append(temp)
                        except ValueError:
                            new_data.append(feature_averages[feature])
            x.append(new_data)
    return x

dataset_dir = 'datasets' # Folder containing data sets
save_dir = 'saved_model' # Folder containing trained model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived']


if __name__=='__main__':    
    # Get feature average to substitute for missing samples
    feature_averages = getAverages(os.path.join(dataset_dir, 'train.csv'), ['Age'])
    
    # Import training data
    data = getFeatures(os.path.join(dataset_dir, 'train.csv'), features, feature_averages)
    trX = np.array(data)[:, 0:6]
    trY = np.array(data)[:, 6:7]

    # Create the model
    feature_size = len(features)-1
    x = tf.placeholder(tf.float32, [None, feature_size])
    W = tf.Variable(tf.zeros([feature_size, 1]), name='W')
    b = tf.Variable(tf.zeros([1]), name='b')
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    # Define loss function and optimizer with L2 Regularization
    beta = 0.01
    L2regularizer = tf.nn.l2_loss(W)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y) + beta*L2regularizer) 
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
     
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    # Train
    for step in range(1000):
#         print("Step", step)
        sess.run(train_step, feed_dict={x: trX, y_: trY})
     
    # Save trained model
#     saver.save(sess, os.path.join(save_dir, 'trained_model'))
     
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Training accuracy:", sess.run(accuracy, feed_dict={x: trX, y_: trY}))
    
    # Import test data
    data = getFeatures(os.path.join(dataset_dir, 'test.csv'), features[:-1], feature_averages)
    teX = np.array(data)
    
    # Test
    print("Test result:", sess.run(y, feed_dict={x: trX}))