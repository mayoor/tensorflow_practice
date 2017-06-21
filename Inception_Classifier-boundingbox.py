
# coding: utf-8

# In[26]:


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
slim  = tf.contrib.slim
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import json


# In[27]:


from nets import inception
from preprocessing import inception_preprocessing
import os
from urllib.request import urlopen
from datasets import imagenet
from os import walk


# In[28]:


print (inception.inception_v4.default_image_size)
image_size = inception.inception_v4.default_image_size
checkpoints_dir = '/Users/mayoor/dev/logistic_regression/inception'


# In[29]:


def add_regression_head(bb_net,dropout_keep_prob=0.8):
    dropout_keep_prob=0.8
    with tf.variable_scope('BB_Logits'):
        # 8 x 8 x 1536
        net = slim.avg_pool2d(bb_net, bb_net.get_shape()[1:3], padding='VALID',
                            scope='AvgPool_1a_bb')
        # 1 x 1 x 1536
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_bb')
        net = slim.flatten(net, scope='PreLogitsFlatten_bb')
        net = slim.fully_connected(net, 4, activation_fn=None,
                                    scope='BB_Logits')
    return net


# In[30]:


def load_data(reader_queue):
    label = reader_queue[1]
    image_string = tf.read_file(reader_queue[0])
    print (image_string)
    print (label)
    label_oh = tf.one_hot(label, depth=4)
    image = tf.image.decode_image(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=True)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    return tf.reshape(processed_images,[image_size,image_size,3]), label_oh


# In[31]:


def load_data_bnd(reader_queue):
    label = reader_queue[1]
    image_string = tf.read_file(reader_queue[0])

    label_oh = label
    image = tf.image.decode_image(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=True)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    return tf.reshape(processed_images,[image_size,image_size,3]), label_oh


# In[7]:


with tf.Graph().as_default():
    labels = json.loads(open('super_hero_meta.json','r').read())
    reader_queue = tf.train.string_input_producer(getDataSet('super_heros')['batman'])
    image, label = load_data(reader_queue, labels)
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=50000,
        min_after_dequeue=10000)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        val = sess.run(k)
        print (val)
        coord.request_stop()
        coord.join(threads)


# In[32]:


def getAllFiles(root_folder, data_file):
    data_js = os.path.join(root_folder, data_file)
    bnd_boxes = []
    image_files = []
    with open(data_js,'r') as df:
        data_map = json.loads(df.read())
        all_data = data_map['images']
        for rec in all_data:
            image_files.append(os.path.join(root_folder, rec['location']))
            bnd_boxes.append(np.array([float(rec['bindbox']['xmin']),float(rec['bindbox']['ymin']),float(rec['bindbox']['xmax']),float(rec['bindbox']['ymax'])]))
    return image_files, bnd_boxes


# In[33]:


def getDataSet(root_folder, data_file):
    data = defaultdict(list)
    for (dirpath, dirnames, filenames) in walk(root_folder):
        for image in filenames:
            label = dirpath[dirpath.index('/')+1:]
            image_file = os.path.join(dirpath,image)
            data[label].append(image_file)
    return data


# In[24]:


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=["InceptionV4/Logits", "InceptionV4/AuxLogits"]
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = ['BB_Logits']
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    #variables_to_restore = []
    #for var in slim.get_model_variables():
    #    variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
      variables_to_restore)


# In[36]:


def get_new_init_fn():
    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
        slim.get_model_variables('InceptionV4'))


# In[37]:


batch_size = 10
with tf.Graph().as_default():
    #labels = json.loads(open('super_hero_meta.json','r').read())
    #reverse_labels = json.loads(open('super_hero_meta_reverse.json','r').read())
    #reader_queue = tf.train.string_input_producer(getAllFiles('super_heros'))
    i, bnd = getAllFiles('/Users/mayoor/dev/slim/VOCtrainval_11-May-2012/VOCdevkit/VOC2012', 'voc_annot.json')
    igs = tf.convert_to_tensor(i, dtype=tf.string)
    bnds = tf.stack(bnd)
    print (bnds.get_shape())
    reader_queue = tf.train.slice_input_producer([igs,bnds],
                                            num_epochs=2,
                                            shuffle=True)
    image, label = load_data_bnd(reader_queue)
    processed_images, bb_target = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=50000,
        min_after_dequeue=10000)
    
    print(processed_images.get_shape())
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        logits, endpoints = inception.inception_v4(processed_images, num_classes=1001, is_training=True)
        bb_net = endpoints['Mixed_7d']
    
    probabilities = tf.nn.softmax(logits)
    bb_logits = add_regression_head(bb_net)
    print('Added regression head.....')

    #init_fn, init_dict = slim.assign_from_checkpoint(
    #    os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
    #    slim.get_model_variables('InceptionV4'))
    print('Inception weights loaded.....')
    
    print (bb_logits, bb_target)
    loss = slim.losses.mean_pairwise_squared_error(bb_logits, bb_target)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    total_loss = slim.losses.get_total_loss()
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    logdir = 'incep_bb'
    print ("Starting training..")
    slim.learning.train(train_op,logdir,init_fn=get_new_init_fn(),number_of_steps=2,save_summaries_secs=100,save_interval_secs=600)
    

