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
from PIL import Image, ImageDraw, ImageFont
input_video="../test_video/8.mp4"
modeldir = './model/20170511-185253.pb'
classifier_filename = './model/classifier.pkl'
npy='./npy'
train_img="../train_img"
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

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

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture = cv2.VideoCapture(input_video)
        c = 0
        total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # 帧率
        print(fps)
        print(int(total / fps))  # 视频时间
        print(total)
        print('Start Recognition')

        cc = np.zeros((10, 4), dtype=np.int32)
        count = 0
        name = []
        prevTime = 0
        num=0
        while True:
            ret, frame = video_capture.read()
            start = time.time()
            # long running
            # do something other
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            curTime = time.time()+1    # calc fps
            timeF = frame_interval
            if (c % timeF == 0) :
                count = 0
                name=[]
                find_results = []
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)
                if nrof_faces > 0 :
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)
                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        cc[i][0] = det[i][0]
                        cc[i][1] = det[i][1]
                        cc[i][2] = det[i][2]
                        cc[i][3] = det[i][3]
                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue
                        try:
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))

                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            #print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            # print("predictions")
                            print(best_class_indices,' with accuracy ',best_class_probabilities)

                            # print(best_class_probabilities)
                            if best_class_probabilities>0.4:
                                count+=1
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                #plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                print('Result Indices: ', best_class_indices[0])
                                print(HumanNames)

                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]

                                        name.append(result_names)
                                        frame = cv2ImgAddText(frame, result_names, text_x, text_y)
                                        # cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        #              1, (0, 0, 255), thickness=1, lineType=2)

                        except:
                            pass
                else:
                    print('Alignment Failure')
            # c+=1
            num+=1
            # for i in range(count):
            #     print(i)
            #     print(name[i])
            #     cv2.rectangle(frame, (cc[i][0], cc[i][1]), (cc[i][2], cc[i][3]), (0, 255, 0), 2)
            #     frame = cv2ImgAddText(frame, name[i], cc[i][0], cc[i][3] + 20)
            end = time.time()
            print(end - start, num)
            cv2.imshow('Video', frame)
            #cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()