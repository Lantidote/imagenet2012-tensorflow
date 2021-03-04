import os
import random
import json
import tensorflow as tf
import cv2
from multiprocessing import Process

IMAGE_DIR = './data'
TFRECORD_DIR = './tfrecord/'
random.seed(0)
cores = 4
image_per_record = 100


def get_dataset_dict(imagedir):
    root_dir = imagedir
    category = [x[1] for x in os.walk(imagedir)][0]
    dataset = {}
    label = {}
    for j, class_name in enumerate(category):
        subdir = os.path.join(root_dir, class_name)
        imagelist = os.listdir(subdir)
        random.shuffle(imagelist)
        label[class_name] = j
        train_dataset = imagelist[:1000]
        test_dataset = imagelist[1000:]
        print(class_name, 'train nums:', str(len(train_dataset)), 'test nums:', str(len(test_dataset)))
        dataset[class_name] = {
            'dir': subdir,
            'train': train_dataset,
            'valid': test_dataset
        }
    return dataset, label


def process_in_queues(dataset, label, dataset_type='train'):
    tfrecord_dir = TFRECORD_DIR + dataset_type + '_tf'
    total_class_num = len(dataset)
    each_process_class_num = int(total_class_num / cores)
    files_for_process_list = []
    for i in range(cores - 1):
        files_for_process_list.append(
            list(dataset.items())[i * each_process_class_num:(i + 1) * each_process_class_num])
    files_for_process_list.append(list(dataset.items())[(cores - 1) * each_process_class_num:])

    # create_tfrecord(dict(files_for_process_list[0]), label, tfrecord_dir, dataset_type)  # 单线程
    processes_list = []
    for i in range(cores):
        processes_list.append(Process(target=create_tfrecord,
                                      args=(dict(files_for_process_list[i]), label,
                                            tfrecord_dir, dataset_type), daemon=True))

    for p in processes_list:
        p.start()
    for p in processes_list:
        p.join()  # 子线程全部加入，主线程等所有子线程运行完毕

    print(dataset_type, 'end')


def create_tfrecord(dataset, label, tfrecord_dir, dataset_type='train'):
    for class_name, info in dataset.items():
        record_path = os.path.join(tfrecord_dir,
                                   '{}_{}_{:0>3d}.tfrecords'.format(dataset_type, class_name, 0))
        writer = tf.io.TFRecordWriter(record_path)
        for i in range(len(info[dataset_type])):
            if i % image_per_record == 0:
                writer.close()
                record_path = os.path.join(tfrecord_dir,
                                           '{}_{}_{:0>3d}.tfrecords'.format(dataset_type, class_name,
                                                                            i // image_per_record))
                print(dataset_type, class_name, str(i // image_per_record), 'started')
                writer = tf.io.TFRecordWriter(record_path)
            example = create_tfrecord_example(class_name, os.path.join(info['dir'], info[dataset_type][i]), label)
            writer.write(example.SerializeToString())
        writer.close()


def create_tfrecord_example(class_name, image_file, label):
    image = cv2.imread(image_file)
    bytes_image = cv2.imencode('.jpg', image)[1].tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': int64_feature(label[class_name]),
        'image': bytes_feature(bytes_image)
    }))
    return example


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    dataset, label = get_dataset_dict(IMAGE_DIR)
    num = 0
    for key, value in dataset.items():
        num += len(value['valid'])
    print("validation nums ==>", num)
    j_d = json.dumps(dataset)
    j_l = json.dumps(label)
    with open('j_d.json', 'w', encoding='utf-8') as f:
        f.write(j_d)
        f.close()
    with open('j_l.json', 'w', encoding='utf-8') as f:
        f.write(j_l)
        f.close()
    process_in_queues(dataset, label, 'train')
    process_in_queues(dataset, label, 'valid')
