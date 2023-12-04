#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from keras.models import load_model

class ImageBoundingBoxOverlay:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, self.image_callback)
        self.bbox_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)
        self.image_pub = rospy.Publisher("/overlayed_image", Image, queue_size=10)
        self.current_image = None
        self.keras_model = load_model('path_to_your_model.h5')  # Update with the path to your Keras model

    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)

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

if __name__ == '__main__':
    rospy.init_node('image_bbox_overlay')
    ibo = ImageBoundingBoxOverlay()
    rospy.spin()
