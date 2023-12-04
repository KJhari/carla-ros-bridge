#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from keras.models import load_model

class TrafficSignClassifier:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/darknet_ros/detection_image", Image, self.image_callback)
        self.bbox_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)
        self.keras_model = load_model('./input/traffic-signs-classification-with-cnn/model-23x23.h5')  # Update this
        self.image_pub = rospy.Publisher("/overlayed_image", Image, queue_size=10)
        self.current_image = None
        self.labels = pd.read_csv('./input/traffic-signs-preprocessed/label_names.csv')  # Update this


    def bbox_callback(self, data):
        if self.current_image is not None:
            for box in data.bounding_boxes:
                cropped_img = self.current_image[box.ymin:box.ymax, box.xmin:box.xmax]
                # Preprocess cropped_img for Keras model prediction
                # ...
                prediction = self.keras_model.predict(preprocessed_img)
                label = prediction.argmax(axis=-1)  # Update with your model's specific prediction method
                
                # Overlay bounding box and label onto current_image
                cv2.rectangle(self.current_image, 
                              (box.xmin, box.ymin), 
                              (box.xmax, box.ymax), 
                              (0, 255, 0), 2)
                cv2.putText(self.current_image, 
                            str(label), 
                            (box.xmin, box.ymin - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.current_image, "bgr8"))
            except Exception as e:
                rospy.logerr(e)


    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        mean_image_path = './input/traffic-signs-preprocessed/mean_image_rgb.pickle'
        # Preprocess the image for Keras model
        # This needs to be adapted based on your model's preprocessing requirements
        preprocessed_image = self.preprocess_for_keras(cv_image,mean_image_path)

        # Classify the image
        predictions = self.keras_model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=-1)
        label = self.labels.iloc[predicted_class]['SignName']

        # You can now use the label as needed, for example:
        rospy.loginfo('Detected traffic sign: {}'.format(label))

    def preprocess_for_keras(cv_image, mean_image_path):
        # Load the mean image
        with open('mean_image_path', 'rb') as f:
            mean = pickle.load(f, encoding='latin1')  # dictionary type

        # Assuming cv_image is already read using cv2
        # Resize the image to match the input size of the model (e.g., 32x32)
        preprocessed_image = cv2.resize(cv_image, (32, 32))

        # Convert to RGB
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

        # Subtract the dataset mean
        preprocessed_image = preprocessed_image - mean['mean_image_rgb']

        # Scale pixel values to [0, 1]
        preprocessed_image = preprocessed_image.astype(np.float32) / 255.0

        # Add a fourth dimension for batch size (as Keras expects)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        return preprocessed_image

if __name__ == '__main__':
    rospy.init_node('traffic_sign_classifier', anonymous=True)
    tsc = TrafficSignClassifier()
    rospy.spin()
