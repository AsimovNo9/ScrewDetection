import os

os.add_dll_directory("C:/Program Files/Nvidia GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


class Prepare:
    def __init__(self, modelURL):

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

    def annotate(self):
        labels = [{"name": "screw", "id": 1}]

        with open(self.annotation_path + "\label_map.pbtxt", "w") as f:
            for label in labels:
                f.write("item{\n")
                f.write("\tname:'{}'\n".format(label["name"]))
                f.write("\tid:{}\n".format(label["id"]))
                f.write("}\n")

        command_train = f"python {self.scripts_path}/generate_tfrecord.py -x {self.image_path}/train -l {self.annotation_path}/label_map.pbtxt -o {self.annotation_path}/train.record"
        os.system(command_train)
        command_test = f"python {self.scripts_path}/generate_tfrecord.py -x {self.image_path}/test -l {self.annotation_path}/label_map.pbtxt -o {self.annotation_path}/test.record"
        os.system(command_test)

    def download_model(self):

        os.makedirs(self.pretrained_model_path, exist_ok=True)

        get_file(
            fname=self.filename,
            origin=self.modelURL,
            cache_dir=self.pretrained_model_path,
            extract=True,
        )

    def update_config(self):
        config = config_util.get_configs_from_pipeline_file(self.config_path)
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = 1
        pipeline_config.train_config.batch_size = 2
        pipeline_config.train_config.fine_tune_checkpoint = (
            self.pretrained_model_path
            + "/datasets/"
            + self.modelName
            + "/checkpoint/ckpt-7"
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
            self.annotation_path + "/train.record"
        ]
        pipeline_config.eval_input_reader[0].num_epochs = 3
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = (
            0.005
        )
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = (
            0.001
        )

        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(self.config_path, "wb") as f:
            f.write(config_text)

    def train_model(self):

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        command = f"python {self.apimodel_path}/research/object_detection/model_main_tf2.py --model_dir={self.model_path}/{self.modelName} --pipeline_config_path={self.model_path}/{self.modelName}/pipeline.config --alsologtostderr --num_train_steps=12000"
        os.system(command)

    def evaluate(self):
        eval_commmand = f"tensorboard --logdir={self.model_path}"
        os.system(eval_commmand)
