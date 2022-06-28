from scripts.set import *

"""
Run each line of code according to your needs and your progress with the previous line

"""

# Call the function to prepare the model and pass the link of any model as a string argument
model = Prepare(
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
)
model.download_model()  # Downloads model corresponding to link passed as argument

model.annotate()  # Creates label and record files

model.update_config()  # Updates the pipeline confifuration file in the model folder of the desired model

model.train_model()  # Train the model using the pipeline configuration file to derive new checkpointsweights

# model.evaluate()  # evaluate the metrics of the model

# detection_model = model.load()  # Load the model and save in a  variable detection_model

# model.realtime_detect(
#     detection_model
# )  # Start real time detection using the loaded model
