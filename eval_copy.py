from scripts.set import *

"""
Run each line of code according to your needs and your progress with the previous line

"""

# Call the function to prepare the model and pass the link of any model as a string argument
model = Prepare(
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
)

model.evaluate()  # evaluate the metrics of the model
