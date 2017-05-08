# coding:utf-8

import os
import glob
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.platform import gfile

#Inception-V3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量岁对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '../models'
MODEL_FILE = 'classify_image_graph_def.pb'

CACHE_DIR = '../tmp/bottleneck'

INPUT_DATA = '../data/flower_photos'
FEATURE_DIR = '../data/features'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# CNN超参数
LEARNING_RATE = 0.01
STEPS = 3000
BATCH_SIZE = 64


# 此函数从数据文件夹中读取所有的图片列表病按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    res = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # sub_dirs: ['../data/flower_photos', '../data/flower_photos/tulips', '../data/flower_photos/sunflowers', '../data/flower_photos/roses']
    # 第一个是当前目录，予以删除
    del sub_dirs[0]
    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        filelist = []
        # 这里dir_name就是花的名字：tulips, sunflowers, roses
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            filelist.extend(glob.glob(file_glob))
        label_name = dir_name.lower()
        train_images = []
        vali_images = []
        test_images = []
        for filepath in filelist:
            filename = os.path.basename(filepath)
            chance = np.random.randint(100)
            if chance < testing_percentage:
                test_images.append(filename)
            elif chance < (testing_percentage + validation_percentage):
                vali_images.append(filename)
            else:
                train_images.append(filename)
        res[label_name] = {
            'dir': dir_name,
            'train': train_images,
            'vali': vali_images,
            'test': test_images,
        }
    """
    key:  tulips
    dir =  tulips
    train.count =  396
    vali.count =  119
    test.count =  118
    
    key:  sunflowers
    dir =  sunflowers
    train.count =  411
    vali.count =  128
    test.count =  160
    
    key:  roses
    dir =  roses
    train.count =  390
    vali.count =  136
    test.count =  115
    """
    return res


# 此函数通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists：create_image_lists函数返回数据，保存了所有的数据
# image_dir：根目录
# label_name：类别名称，‘tulips’， ‘sunflowers’， ‘roses’
# index：图片编号
# category：'train', 'vali', 'test'
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 该函数通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category)


# 该函数使用加载好的Inception-v3模型处理一张图片，得到这张图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 该函数获取一张图片经过 Inception-v3 模型处理之后的特征向量，它会先试图寻找已经计算好且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 先获取一张图片对应的特征向量文件的地址
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果特征向量文件不存在，则通过 Inception-v3模型来计算特征向量， 并将计算的结果存入文件。
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_str = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_str)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_str = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_str.split(',')]
    return bottleneck_values


# 随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 获取全部的测试数据，最终测试时，要在所有的数据上测试
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'test'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 获取所有图片的经过Inception-v3处理得到的特征向量
def get_bottleneck_features(sess, image_data_tensor, bottleneck_feature_tensor):
    if not os.path.exists(FEATURE_DIR):
        image_subdirs = [x[0] for x in os.walk(INPUT_DATA)]
        del image_subdirs[0]
        for image_subdir in image_subdirs:
            class_name = os.path.basename(image_subdir)
            feature_subdir = os.path.join(FEATURE_DIR, class_name)
            os.makedirs(feature_subdir)
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
            image_paths = []
            for extension in extensions:
                image_glob = os.path.join(image_subdir, '*.' + extension)
                image_paths.extend(glob.glob(image_glob))
            for image_path in image_paths:
                print image_path
                feature_path = os.path.join(feature_subdir, os.path.basename(image_path))
                image_data = gfile.FastGFile(image_path, 'rb').read()
                if image_data:
                    bottleneck_feature = sess.run(bottleneck_feature_tensor, {image_data_tensor: image_data})
                    bottleneck_feature = np.squeeze(bottleneck_feature)
                    with open(feature_path, 'w') as bottleneck_file:
                        bottleneck_str = ','.join(str(x) for x in bottleneck_feature)
                        bottleneck_file.write(bottleneck_str)
                else:
                    print 'image bad'
    feature_list = {}
    feature_list['train'] = []
    feature_list['vali'] = []
    feature_list['test'] = []
    feature_list['class_name'] = []
    feature_subdirs = [x[0] for x in os.walk(FEATURE_DIR)]
    del feature_subdirs[0]
    for feature_subdir in feature_subdirs:
        class_name = os.path.basename(feature_subdir)
        feature_list['class_name'].append(class_name)
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
        feature_paths = []
        for extension in extensions:
            image_glob = os.path.join(feature_subdir, '*.' + extension)
            feature_paths.extend(glob.glob(image_glob))
        for feature_path in feature_paths:
            r = random.randrange(100)
            if r < 10:
                feature_list['vali'].append(feature_path)
            elif r < 20:
                feature_list['test'].append(feature_path)
            else:
                feature_list['train'].append(feature_path)
    np.random.shuffle(feature_list['train'])
    np.random.shuffle(feature_list['vali'])
    np.random.shuffle(feature_list['test'])
    print 'train.count:', len(feature_list['train'])
    print 'vali.count:', len(feature_list['vali'])
    print 'test.count:', len(feature_list['test'])
    return feature_list


# 根据feature_path得到feature
def get_feature_with_feature_path(feature_path):
    with open(feature_path, 'r') as bottleneck_file:
        bottleneck_str = bottleneck_file.read()
    bottleneck_feature = [float(x) for x in bottleneck_str.split(',')]
    return bottleneck_feature

def get_y_with_feature_path(feature_list, feature_path):
    class_name = feature_path.split('/')[-2]
    keys = feature_list['class_name']
    y = np.zeros([len(keys)])
    y[keys.index(class_name)] = 1.0
    return y

# 根据epoch获取训练batch
def get_train_features(feature_list, epoch):
    train_feature_paths = feature_list['train']
    start = epoch % (len(train_feature_paths) / (BATCH_SIZE+1))
    batch_feature_paths = train_feature_paths[start: start+BATCH_SIZE]
    batch_x = []
    batch_y = []
    for feature_path in batch_feature_paths:
        bottleneck_feature = get_feature_with_feature_path(feature_path)
        bottleneck_y = get_y_with_feature_path(feature_list, feature_path)
        batch_x.append(bottleneck_feature)
        batch_y.append(bottleneck_y)
    return batch_x, batch_y

def get_vali_feature(feature_list):
    vali_x = []
    vali_y = []
    for feature_path in feature_list['vali']:
        bottleneck_feature = get_feature_with_feature_path(feature_path)
        bottleneck_y = get_y_with_feature_path(feature_list, feature_path)
        vali_x.append(bottleneck_feature)
        vali_y.append(bottleneck_y)
    return vali_x, vali_y

def get_test_feature(feature_list):
    test_x = []
    test_y = []
    for feature_path in feature_list['test']:
        bottleneck_feature = get_feature_with_feature_path(feature_path)
        bottleneck_y = get_y_with_feature_path(feature_list, feature_path)
        test_x.append(bottleneck_feature)
        test_y.append(bottleneck_y)
    return test_x, test_y


def main():
    # 获取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取 Inception-v3模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], 'BottleneckInputPlaceholder')
    groud_truth_input = tf.placeholder(tf.float32, [None, n_classes], 'GroundTruthInput')
    # 定义一层全连接层解决新的图片分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=groud_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(groud_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feature_list = get_bottleneck_features(sess, jpeg_data_tensor, bottleneck_tensor)
        for i in range(STEPS):
            # 获取一个batch的训练数据
            # train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'train', jpeg_data_tensor, bottleneck_tensor)
            batch_train_x, batch_train_y = get_train_features(feature_list, i)
            sess.run(train_step, feed_dict={bottleneck_input: batch_train_x, groud_truth_input: batch_train_y})
            if i % 100 == 0 or i+1 == STEPS:
                # vali_bottlenecks, vali_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'vali', jpeg_data_tensor, bottleneck_tensor)
                vali_x, vali_y = get_vali_feature(feature_list)
                vali_acc, vali_loss = sess.run([evaluation_step, cross_entropy_mean], feed_dict={bottleneck_input: vali_x, groud_truth_input: vali_y})
                print 'Step:', i, ', vali loss =', vali_loss, ', vali_acc =', vali_acc
        print 'Optimization Finished!'
        # test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_x, test_y = get_test_feature(feature_list)
        test_loss, test_acc = sess.run([cross_entropy_mean, evaluation_step], feed_dict={bottleneck_input: test_x, groud_truth_input: test_y})
        print '\nTest loss =', test_loss, ', Test_acc =', test_acc

if __name__ == '__main__':
    main()








