import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
import numpy as np
from dbac_lib import vgg_preprocessing
import abc
import logging

logger = logging.getLogger(__name__)

FEAT_TYPE = ['vgg16', 'mem_cached']
MODELS_PATH = ['/home/rfsc/Projects/dbac/data/vgg_16.ckpt']


class IFeatureExtractor(metaclass=abc.ABCMeta):
    def __init__(self, feat_type):
        self.name = feat_type

    @staticmethod
    def factory(feat_type, **kwargs):
        if feat_type == FEAT_TYPE[0]:
            return VGG16Features(feat_type, **kwargs)
        elif feat_type == FEAT_TYPE[1]:
            return MemCachedFeatures(feat_type, **kwargs)
        else:
            raise ValueError("There is no {} feature extraction type. Try {}.".format(feat_type, FEAT_TYPE))

    @abc.abstractclassmethod
    def load(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def compute(self, images_path):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def release(self):
        raise NotImplementedError()


class VGG16Features(IFeatureExtractor):
    def __init__(self, feat_type, **kwargs):

        # set parameters
        super().__init__(feat_type)
        self.weights_file = kwargs.get('weights_file', MODELS_PATH[0])
        self.feat_name = kwargs.get('feat_name', 'vgg_16/fc7/BiasAdd:0')
        logger.info("VGG16 feature extractor set up with model {} for compute feature {}."
                    .format(self.weights_file, self.feat_name))

        # Placeholders
        self.init_fn_op = None
        self.input_tensor = None
        self.features_tensor = None
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.graph, config=config)

        # Define model and load weights
        with self.graph.as_default() as graph:
            self.input_tensor = tf.placeholder(tf.string, name='feat_ext/input')
            image = tf.image.decode_jpeg(tf.read_file(self.input_tensor), channels=3)
            processed_image = vgg_preprocessing.preprocess_image(image, 224, 224, is_training=False)
            processed_images = tf.expand_dims(processed_image, 0)
            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
                feats = graph.get_tensor_by_name(self.feat_name)
                self.features_tensor = tf.squeeze(feats)
                self.init_fn_op = slim.assign_from_checkpoint_fn(self.weights_file, slim.get_model_variables('vgg_16'))

    def load(self):
        logger.info("Loading VGG16 feature extractor.")
        self.init_fn_op(self.tf_session)

    def compute(self, images_path):
        feats_cache = []
        for img_path in [images_path] if isinstance(images_path, str) else images_path:
            feats = self.tf_session.run(self.features_tensor, feed_dict={self.input_tensor: img_path})
            feats_cache.append(feats)
        return np.array(feats_cache)

    def release(self):
        logger.info("Releasing VGG16 feature extractor resources.")
        self.tf_session.close()


class MemCachedFeatures(IFeatureExtractor):

    def __init__(self, feat_type, **kwargs):
        super().__init__(feat_type)
        self.file = kwargs.get('cache_file')
        logger.info("Memory cached feature extractor set up for cache file {}".format(self.file))
        self.feat_dic = None

    def load(self):
        logger.info("Loading feature cache to memory.")
        self.feat_dic = np.load(self.file).item()

    def release(self):
        logger.info("Releasing feature cache from memory.")
        del self.feat_dic

    def compute(self, key):
        feats_cache = []
        for img_path in [key] if isinstance(key, str) else key:
            feats_cache.append(self.feat_dic[img_path])
        feats_cache = np.vstack(feats_cache)
        return feats_cache
