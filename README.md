# Small-Object-Detection-An-Image-Tiling-Based-Approach

### Original images in data folder can be found [here](https://downloads.greyc.fr/vedai/)

# Abstract
Computer vision is a challenging task as machine sees numbers, unlike we humans. The computer has to detect and classify objects present in an image to perceive as we humans do. Most of the popular object detectors like RCNN-family or YOLO-family detect medium and large objects well but they find difficulty in detecting small objects. Most of these detectors use Convolutional Neural Networks to extract features for Object localization and Object Classification. The features of smaller objects may disappear in deeper layers and it becomes difficult for the detector to detect small objects. One solution is to use high-resolution images for small objects detection. But training models with high-resolution images will be slow and needs huge GPU memory. The inference time is also more for high-resolution images. To overcome this difficulty we have used an image tiling-based approach to detect small objects. Custom YOLOv4 (small) is used for transfer learning and detections are performed on CPU for performance evaluation. The metric used for evaluation is mAP. Finally, the model is deployed in the local cloud and a GUI is developed to do object detections locally.

# [Link](https://binginagesh.medium.com/small-object-detection-an-image-tiling-based-approach-bce572d890ca) to medium blog.


# Sample predictions in local cloud
https://user-images.githubusercontent.com/46963154/137170709-17c4acda-1842-4fd5-9b56-8b5a1a5203b3.mp4


# Sample predictions on GUI
https://user-images.githubusercontent.com/46963154/137169434-9bc7c84f-262d-495a-abd2-b3e1d98905c6.mp4

# Conclusion
Computer Vision is a challenging task because computers see images as numbers. CNNs are at the heart of modern object detection algorithms. The features of smaller objects may disappear in deeper layers and it becomes difficult for the detector to detect small objects. Experiments are done with YOLO v4 architecture. The best YOLO v4 model is converted to TensorFlow but it takes significantly more time when compared to Darknet. This is expected as Darknet is run directly in C whereas TensorFlow has Python wrappers around C/C++ code. TensorFlow model is converted to TensorFlow-Lite version to run on low compute Edge devices. The model is deployed in the local environment and a demo video is shown. For people who don't want to compromise their data a GUI is developed, which can be run on the local instance like a computer, low compute edge devices, etc.

# References
[Image tiling](https://github.com/nagi1995/yolo-tiling)

[Darknet](https://github.com/nagi1995/darknet)

[YOLO to TF and TF-Lite conversion](https://github.com/nagi1995/tensorflow-yolov4-tflite)
