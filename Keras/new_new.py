#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import pickle
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from keras.models import load_model
import pandas as pd
import os

class TrafficSignRecognition:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/darknet_ros/detection_image", Image, self.image_callback)
        self.bbox_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)
        self.image_pub = rospy.Publisher("/traffic_sign_recognition/image_with_boxes", Image, queue_size=10)
        self.current_image = None
        self.bounding_boxes = []
        self.mean = None

        # Load model
        model_path = '/home/kj/carla-ros-bridge/Keras/input/traffic-signs-classification-with-cnn/model-23x23.h5'
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            rospy.logerr(f"Model file not found: {model_path}")

        mean_image_path = '/home/kj/carla-ros-bridge/Keras/input/trafficSignsPreprocessed/mean_image_rgb.pickle'
        if os.path.exists(mean_image_path):
            print('found mean pickle model')
        else:
            rospy.logerr(f"Model file not found: {mean_image_path}")
        
        try:
            with open(mean_image_path, 'rb') as f:
                self.mean = pickle.load(f, encoding='latin1')  # Load the file contents once
            print('working ...')
            print(self.mean['mean_image_rgb'].shape)  # Correctly access the 'mean_image_rgb' key from self.mean
        except FileNotFoundError:
            rospy.logerr(f"Mean image file not found: {mean_image_path}")
            self.mean = None  # Set to None or handle appropriately
        except Exception as e:
            rospy.logerr(f"An error occurred while loading the mean image: {e}")
            self.mean = None  # Set to None or handle appropriately

        
        # Load labels
        self.labels = pd.read_csv('/home/kj/carla-ros-bridge/Keras/input/trafficSignsPreprocessed/label_names.csv')

    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.process_image()  # Call process_image here
        except Exception as e:
            rospy.logerr(e)

    def bbox_callback(self, data):
        self.bounding_boxes = data.bounding_boxes
        if self.current_image is not None:
            self.process_image()  # Call process_image here if image is already received

    def process_image(self):
        if self.current_image is not None and self.bounding_boxes and self.mean is not None:
            for box in self.bounding_boxes:
                # Extracting the bounding box
                x_min, y_min = box.xmin, box.ymin
                box_width, box_height = box.xmax - box.xmin, box.ymax - box.ymin
                c_ts = self.current_image[y_min:y_min+box_height, x_min:x_min+box_width]

                if c_ts.size == 0:
                    continue

                # Preprocess for Keras model
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                blob_ts[0] = blob_ts[0, :, :, :] - self.mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)

                # Prediction
                scores = self.model.predict(blob_ts)
                prediction = np.argmax(scores)
                predicted_label = self.labels['SignName'][prediction]

                # Draw bounding box and label on the image
                cv2.rectangle(self.current_image, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)
                cv2.putText(self.current_image, predicted_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert the processed image back to a ROS Image message
            try:
                ros_msg = self.bridge.cv2_to_imgmsg(self.current_image, "bgr8")
                # Publish the ROS Image message to the new topic
                self.image_pub.publish(ros_msg)
            except Exception as e:
                rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('traffic_sign_recognition')
    tsr = TrafficSignRecognition()
    rospy.spin()
