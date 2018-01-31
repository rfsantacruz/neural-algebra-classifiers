import logging
import os
import random
import re
import threading
import time

import itertools
import tensorflow as tf

logger = logging.getLogger(__name__)


class CycleIterator:
    def __init__(self, items, shuffle=True, max_cycles=None):
        """
        Cycle iterator around a list with shuffle option
        :param items: items to iterate over
        :param shuffle: if true, shuffle on each cycle
        :param max_cycles: number max of cycles. If None iterate indefinitely
        :return: iterator
        """

        # options
        assert items, "Items must be a non-empty iterable"
        self._items = items
        self._shuffle = shuffle
        self._index = list(range(len(items)))
        if shuffle:
            random.shuffle(self._index)

        # Engine attributes
        self._lock = threading.Lock()
        self._current = 0
        self._max_cycles = max_cycles

    def _reset(self):
        self._current = 0
        # reduce cycle
        if not (self._max_cycles is None):
            self._max_cycles -= 1
        # shuffle index
        if self._shuffle:
            random.shuffle(self._index)

    def __iter__(self):
        return self

    def __next__(self):
        # thread safety
        with self._lock:
            if self._max_cycles is None or self._max_cycles > 0:
                item = self._items[self._index[self._current]]
                self._current += 1
                # reset iterator
                if self._current >= len(self._index):
                    self._reset()
                return item
            else:
                raise StopIteration()


def init_logging(file_path='./log_default.txt'):
    """
    Initialize python logging module to log on console and file
    :param file_path: File path to save the log file
    """
    # Log format
    log_formatter = logging.Formatter('[%(asctime)-15s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s')

    # File and console handler
    log_file_handler = logging.FileHandler(file_path, mode='w')
    log_file_handler.setFormatter(log_formatter)
    log_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_file_handler)
    logger.addHandler(console_handler)


def config_tf_session(cuda_vdev_str, mem_groth=True, mem_frac=1.0, graph=None):
    """
    Initialize tensorflow session to be used in Keras.
    >> import keras.backend.tensorflow_backend as KTF
    >> KTF.set_session(config_tf_session('2,3', True, 1.0))
    :param cuda_vdev_str: String for CUDA_VISIBLE_DEVICES enviroment variables
    :param mem_groth: Allow the GPU memory be allocated on demand
    :param mem_frac: Fraction of GPU memory to be used
    :return: Tensorflow session object
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_vdev_str
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = mem_groth
    config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    return tf.Session(graph=graph, config=config)


class TicToc:
    """
    TicToc class for time pieces of code.
    """

    def __init__(self):
        self._TIC_TIME = {}
        self._TOC_TIME = {}

    def tic(self, tag=None):
        """
        Timer start function
        :param tag: Label to save time
        :return: current time
        """
        if tag is None:
            tag = 'default'
        self._TIC_TIME[tag] = time.time()
        return self._TIC_TIME[tag]

    def toc(self, tag=None, fmt=False):
        """
        Timer ending function
        :param tag: Label to the saved time
        :param fmt: if True, formats time in H:M:S, if False just seconds.
        :return: elapsed time
        """
        if tag is None:
            tag = 'default'
        self._TOC_TIME[tag] = time.time()

        if tag in self._TIC_TIME:
            d = (self._TOC_TIME[tag] - self._TIC_TIME[tag])

            if fmt:
                print("Elapsed time is: {} - {}".format(tag, time.strftime('%H:%M:%S', time.gmtime(d))))
            else:
                print("Elapsed time is: {} - {:f} seconds".format(tag, d))

            return d
        else:
            print("No tic() start time available for tag {}.".format(tag))

    # Timer as python context manager
    def __enter__(self):
        self.tic('CONTEXT')

    def __exit__(self, type, value, traceback):
        self.toc('CONTEXT')


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    """
    List pictures in a given directory
    :param directory: directory path
    :param ext: allowed image formats
    :return: list of image file names
    """
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files if re.match(r'(.*\.(?:' + ext + '))', f)]


def get_kwargs_dic(kwargs_str):
    """
    Convert a kwarg string to a dictionary of key-values pairs.
    :param kwargs_str: key vale arguments in str format. e.g. 'learning_rate=1e-2; feat_name=vgg16:add; ...'
    :return: Dictionary of key-value pairs. The keys and values are str and should be processed by its owner.
    """
    kwargs_dic = dict()
    if kwargs_str:
        kwargs_dic = [kv.strip() for kv in kwargs_str.split(';')]
        kwargs_dic = {kv.split('=')[0].strip(): kv.split('=')[1].strip() for kv in kwargs_dic}
    return kwargs_dic


def batch_iterator(n, iterable, fillvalue=None):
    """
    Produces batches of size n over the elements in iterable.
    :param n: batch_size
    :param iterable: items
    :param fillvalue: value to complete las batch
    :return: iterator of batches
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)
