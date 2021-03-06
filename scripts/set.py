import os

import numpy as np
import cv2
import tensorflow as tf
import pyautogui as pya
from tensorflow.python.keras.utils.data_utils import get_file
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from google.protobuf import text_format


class Prepare:
    def __init__(self, modelURL):
        """Prepare constructs the model variables, particularly workspace paths

        Args:
            modelURL (string): Tensorflow model zoo url, or compatible model type from a website
        """

        self.modelURL = modelURL

        self.filename = os.path.basename(modelURL)
        self.modelName = self.filename[: self.filename.index(".")]

        self.workspace_path = "./workspace"
        self.scripts_path = "./scripts"
        self.apimodel_path = "./models"
        self.annotation_path = self.workspace_path + "/annotations"
        self.image_path = self.workspace_path + "/images"
        self.model_path = self.workspace_path + "/models"
        self.pretrained_model_path = self.workspace_path + "/pre-trained-models"
        self.config_path = (
            self.model_path + "/" + f"{self.modelName}" + "/pipeline.config"
        )
        self.checkpoint_path = self.model_path + "/" + f"{self.modelName}"
        self.label_path = self.annotation_path + "/labels.csv"
        self.eval_path = self.checkpoint_path + "/eval"

    def annotate(self):
        """Creates the required annotations, such as the label.pbtxt file, and tf record files"""

        labels = [{"name": "screw", "id": 1}]

        with open(self.annotation_path + "\label_map.pbtxt", "w") as f:
            for label in labels:
                f.write("item{\n")
                f.write("\tname:'{}'\n".format(label["name"]))
                f.write("\tid:{}\n".format(label["id"]))
                f.write("}\n")

        command_valid = f"python {self.scripts_path}/generate_tfrecord.py -x {self.image_path}/valid -l {self.annotation_path}/label_map.pbtxt -o {self.annotation_path}/valid.record"
        os.system(command_valid)
        command_train = f"python {self.scripts_path}/generate_tfrecord.py -x {self.image_path}/train -l {self.annotation_path}/label_map.pbtxt -o {self.annotation_path}/train.record"
        os.system(command_train)
        command_test = f"python {self.scripts_path}/generate_tfrecord.py -x {self.image_path}/test -l {self.annotation_path}/label_map.pbtxt -o {self.annotation_path}/test.record"
        os.system(command_test)

    def download_model(self):
        """Downloads the model using the given link when creating the instance of the model prepare object"""

        os.makedirs(self.pretrained_model_path, exist_ok=True)

        get_file(
            fname=self.filename,
            origin=self.modelURL,
            cache_dir=self.pretrained_model_path,
            extract=True,
        )

        print("file downloaded successfully")

    def update_config(self):
        """Update the pipeline config file for user needs, such as batch number, training steps, number of classes and description of annotation file destinations"""
        config = config_util.get_configs_from_pipeline_file(self.config_path)
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = 1
        pipeline_config.train_config.batch_size = 5
        pipeline_config.train_config.fine_tune_checkpoint = (
            self.pretrained_model_path
            + "/datasets/"
            + self.modelName
            + "/checkpoint/ckpt-0"  # Change based off which checkpoint you are training from
        )
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path = (
            self.annotation_path + "/label_map.pbtxt"
        )
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
            self.annotation_path + "/train.record"
        ]
        pipeline_config.eval_input_reader[0].label_map_path = (
            self.annotation_path + "/label_map.pbtxt"
        )
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
            self.annotation_path + "/valid.record"
        ]
        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(self.config_path, "wb") as f:
            f.write(config_text)

    def train_model(self):
        """Set GPU memory limit and start training model"""

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        command = f"python {self.apimodel_path}/research/object_detection/model_main_tf2.py --model_dir={self.model_path}/{self.modelName} --pipeline_config_path={self.config_path} --alsologtostderr --num_train_steps=25000"
        os.system(command)

    def evaluate(self):
        """Evaluate the model"""
        command = f"python {self.apimodel_path}/research/object_detection/model_main_tf2.py --model_dir={self.model_path}/{self.modelName} --pipeline_config_path={self.config_path} --alsologtostderr --checkpoint_dir={self.checkpoint_path}"
        os.system(command)
        eval_command = f"tensorboard --logdir={self.eval_path}"
        os.system(eval_command)

    def load(self):
        """Load the most recent checkpoint as a model

        Returns:
            detection_model (ckpt): Model that can be used for inference
        """
        configs = config_util.get_configs_from_pipeline_file(self.config_path)
        detection_model = model_builder.build(
            model_config=configs["model"], is_training=False
        )

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(
            os.path.join(self.checkpoint_path, "ckpt-32")
        ).expect_partial()  # Restore based off last created checkpoint
        return detection_model

    def realtime_detect(self, _detection_model):
        """_summary_

        Args:
            _detection_model (ckpt): Create real time detection platform using user screen
        """
        category_index = label_map_util.create_category_index_from_labelmap(
            self.annotation_path + "/label_map.pbtxt"
        )
        detection_model = _detection_model

        while True:

            image = pya.screenshot()

            image_np = np.array(image)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32
            )
            detections = self.detect(input_tensor, detection_model)

            num_detections = int(detections.pop("num_detections"))

            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections["num_detections"] = num_detections
            detections["detection_classes"] = detections["detection_classes"].astype(
                np.int64
            )

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections["detection_boxes"],
                detections["detection_classes"] + label_id_offset,
                detections["detection_scores"],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=0.3,
                agnostic_mode=False,
            )

            cv2.imshow("screw detection", image_np_with_detections)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    @tf.function
    def detect(self, image, detection_model):
        images, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(images, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
