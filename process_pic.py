from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import sys
from PIL import Image, ImageDraw, ImageFont
import shutil

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


img_path = ['1.png','2.png','3.png']
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "../train_img"
input_face="./face"
output_face="./process_face"
class ImageClass():
    def __init__(self,name,image_paths):
        self.name=name
        self.image_paths=image_paths
    def __str__(self):
        return self.name+','+str(len(self.image_paths))+"images"
    def __len__(self):
        return len(self.image_paths)
def get_dataset(paths, has_class_directories=True):
    dataset = []
    for path in paths.split(':'):
        print(path)
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
    return dataset
def get_face_path(paths):
    for path in paths.split(':'):
        path_exp=os.path.expanduser(path)
        actors=os.listdir(path_exp)
        actors.sort()

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
image_data=get_dataset("./actorface")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading feature extraction model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        print('Start Verification!')
        prevTime = 0
        # ret, frame = video_capture.read()
        for faces in image_data:
            i=0
            output_class_dir = os.path.join(output_face, faces.name)
            print(faces.name)
            if os.path.exists(output_class_dir):
                continue
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            for path in "./actorface".split(':'):
                print(path)
                path_exp = os.path.expanduser(path)
                facedir = os.path.join(path_exp, faces.name)
                base_face=os.path.join(facedir, '0.png')
            #frame = cv2.imread(base_face, 0)
            frame = cv2.imdecode(np.fromfile(base_face, dtype=np.uint8), -1)

            curTime = time.time() + 1  # calc fps
            timeF = frame_interval
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]
            emb_array=np.zeros((1, embedding_size))
            scaled = frame
            scaled = cv2.resize(scaled, (input_image_size, input_image_size),
                                interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)

            scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            basic_emb = sess.run(embeddings, feed_dict=feed_dict)
            #print(basic_emb)
            for path  in faces.image_paths:
                #frame = cv2.imread(path, 0)
                frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                curTime = time.time() + 1  # calc fps
                timeF = frame_interval
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                emb_array = np.zeros((1, embedding_size))
                scaled = frame
                scaled = cv2.resize(scaled, (input_image_size, input_image_size),
                                    interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)

                scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                dist = np.linalg.norm(basic_emb - emb_array, ord=2)
                print("dist=", dist)
                if dist < 0.83:
                    shutil.copyfile(path, os.path.join(output_class_dir, str(i) + ".jpg"))
                    i += 1



        # best_class_indices = np.argmax(predictions, axis=1)
        #         # print(best_class_indices)
        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        # print(best_class_probabilities)
        #
        # print(HumanNames)
        # for H_i in HumanNames:
        #     # print(H_i)
        #     if HumanNames[best_class_indices[0]] == H_i:
        #         result_names = HumanNames[best_class_indices[0]]
        #         frame = cv2ImgAddText(frame, result_names, text_x, text_y)
        #         # cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #         #             1, (0, 0, 255), thickness=1, lineType=2)
        #
        # cv2.imshow('Image', frame)
        # if cv2.waitKey(1000000) & 0xFF == ord('q'):
        #     sys.exit("Thanks")
        # cv2.destroyAllWindows()