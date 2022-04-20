import os

from scripts.set import *


preparation = Prepare(
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
)

# preparation.annotate()
# preparation.download_model()
# preparation.update_config()
# preparation.train_model()
preparation.evaluate()
