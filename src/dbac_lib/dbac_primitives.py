import abc
import logging
import numpy as np
from sklearn import svm as sklearn_svm
from sklearn import calibration as sklearn_clb
import pickle
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

PRIMITIVE_TYPES = ['exsvms']


class IPrimitiveCollection(metaclass=abc.ABCMeta):
    def __init__(self, prim_type):
        self.name = prim_type
        # Placeholders
        # array of ids
        self.prim_ids = None
        # array of rprs
        self.prim_rprs = None
        # array of calibrated classifiers
        self.prim_cls = None

    @abc.abstractmethod
    def learn(self, images_path, labels, feat_ext, **kwargs):
        raise NotImplementedError()

    def get_rpr(self, prim_ids):
        return self.prim_rprs[np.where(self.prim_ids == prim_ids)[0][0]]

    def get_ids(self):
        return self.prim_ids

    def get_cls(self, prim_ids):
        return self.prim_cls[np.where(self.prim_ids == prim_ids)[0][0]]

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            self.prim_ids = obj.prim_ids
            self.prim_rprs = obj.prim_rprs
            self.prim_cls = obj.prim_cls

    @staticmethod
    def factory(prim_type, **kwargs):
        if prim_type == PRIMITIVE_TYPES[0]:
            return SVMPrimitives(kwargs)
        else:
            raise ValueError("Primitives of type {} is not supported. Try {}.".format(prim_type, PRIMITIVE_TYPES))


class SVMPrimitives(IPrimitiveCollection):
    def __init__(self, prim_type, **kwargs):
        super().__init__(prim_type)

    def learn(self, images_path, labels, feat_ext, **kwargs):
        # setup parameters
        num_threads = kwargs.get('num_threads', 10)
        ex_size = kwargs.get('ex_size', 10)
        num_ex = kwargs.get('num_ex', 10)
        prim_ids = kwargs.get('prim_ids', np.arange(labels.shape[1]))

        # compute features and train models
        feats = feat_ext.compute(images_path)
        models = Parallel(n_jobs=num_threads)(
            delayed(_train_svm)(feats, labels[:, prim_id], prim_id, ex_size, num_ex) for prim_id in prim_ids)

        # Post Processing
        self.prim_ids = np.array(prim_ids)
        self.prim_rprs, self.prim_cls = [], []
        for idx, prim_id in enumerate(self.prim_ids):
            self.prim_rprs.append(np.vstack([np.hstack((svm_object.intercept_.ravel(), svm_object.coef_.ravel())) for svm_object in models[idx][0]]))
            self.prim_cls.append(models[idx][1])
        self.prim_rprs = np.array(self.prim_rprs)


def _train_svm(feats, labels, prim_id, ex_size, num_ex):
    logger.info("Training Primitive {}.".format(prim_id))

    # split examplars
    pos_img_ids = np.where(labels)[0]
    pos_img_splits = [pos_img_ids] if num_ex == 1 else [pos_img_ids] + [
        np.random.choice(pos_img_ids, size=min(ex_size, pos_img_ids.size), replace=False) for _ in range(num_ex)]
    logger.info("Primitive {} has {} exemplars.".format(prim_id, len(pos_img_splits)))
    svms, clbs = [], []
    for ex_id, pos_ex_ids in enumerate(pos_img_splits):
        if len(pos_ex_ids) > 0:
            logger.info("Primitive {} training exemplar {} ...".format(prim_id, ex_id))
            svm_object = sklearn_svm.LinearSVC(C=1e-3, class_weight={1: 2, -1: 1.0}, verbose=0, penalty='l2',
                                               loss='hinge', dual=True)
            neg_ex_ids = np.array([idx for idx in range(labels.size) if idx not in pos_ex_ids])
            X = np.vstack([feats[pos_ex_ids], feats[neg_ex_ids]])
            Y = np.hstack([np.ones(pos_ex_ids.size), -1.0 * np.ones(neg_ex_ids.size)])
            svm_object.fit(X, Y)
            train_acc = svm_object.score(X, Y)
            svms.append(svm_object)
            logger.info("SVM (Primitive {} examplar {}) has {} positives, {} negatives and accuracy {}."
                        .format(prim_id, ex_id, pos_ex_ids.size, neg_ex_ids.size, train_acc))
            if ex_id == 0:
                svm_object_clb = sklearn_svm.LinearSVC(C=1e-3, class_weight={1: 2, -1: 1.0}, verbose=0, penalty='l2',
                                                   loss='hinge', dual=True)
                np.random.shuffle(pos_ex_ids)
                np.random.shuffle(neg_ex_ids)
                pos_split_point = int(np.ceil(0.9*len(pos_ex_ids)))
                cls_pos_idx, calib_pos_idx = pos_ex_ids[:pos_split_point], pos_ex_ids[pos_split_point:]
                neg_split_point = int(np.ceil(0.9 * len(neg_ex_ids)))
                cls_neg_idx, calib_neg_idx = neg_ex_ids[:neg_split_point], neg_ex_ids[neg_split_point:]
                X = np.vstack([feats[cls_pos_idx], feats[cls_neg_idx]])
                Y = np.hstack([np.ones(cls_pos_idx.size), -1.0 * np.ones(cls_neg_idx.size)])
                svm_object_clb.fit(X, Y)
                clb_object = sklearn_clb.CalibratedClassifierCV(svm_object_clb, cv='prefit')
                X = np.vstack([feats[calib_pos_idx], feats[calib_neg_idx]])
                Y = np.hstack([np.ones(calib_pos_idx.size), -1.0 * np.ones(calib_neg_idx.size)])
                clb_object.fit(X, Y)
                clbs.append(clb_object)
                clb_object.score(X, Y)
                logger.info("Calibrated SVM (Primitive {} examplar {}) has {} positives, {} negatives and accuracy {}."
                            .format(prim_id, ex_id, pos_ex_ids.size, neg_ex_ids.size, train_acc))
    return svms, clbs


class HardMiningSVM:
    def __init__(self, name='default', pos_weight=2.0, c=1e-5, threshold=1.1, limit_retrain=100):
        self.svm_object = sklearn_svm.LinearSVC(
            C=c, class_weight={1: pos_weight, -1: 1.0}, verbose=0, penalty='l2', loss='hinge', dual=True)
        self.pos_feats = None
        self.neg_feats = None
        self.thr = threshold
        self.limit_retrain = limit_retrain
        self.neg_cache_feats = []
        self.name = name
        self.initialized = False
        self.clb_object = sklearn_clb.CalibratedClassifierCV(self.svm_object, method='sigmoid', cv=3)

    def initialize(self, pos_feats):
        self.pos_feats = pos_feats
        self.neg_feats = np.zeros((0, pos_feats.shape[1]), dtype=pos_feats.dtype)
        #self.limit_retrain = np.min([10*self.pos_feats.shape[0], self.limit_retrain])

    def _train(self):
        logger.info("Training SVM model {} for {} positive samples and {} negative samples..."
                    .format(self.name, self.pos_feats.shape[0], self.neg_feats.shape[0]))
        X = np.vstack([self.pos_feats, self.neg_feats])
        Y = np.vstack([np.ones((self.pos_feats.shape[0], 1)), -1.0 * np.ones((self.neg_feats.shape[0], 1))])
        self.svm_object.fit(X, Y.ravel())
        acc = self.svm_object.score(X, Y.ravel())
        self.initialized = True
        logger.info("SVM {} Trained. Accuracy on training {}.".format(self.name, acc))

    def _shrink_neg(self):
        if self.initialized:
            scores = np.multiply(-1.0, self.svm_object.decision_function(self.neg_feats))
            not_easy_index = np.logical_not(scores > self.thr)
            self.neg_feats = self.neg_feats[not_easy_index]

    def _dilate_neg_cache(self, new_feats):
        if self.initialized:
            scores = np.multiply(-1.0, self.svm_object.decision_function(new_feats))
            hard_inds = scores < self.thr
            new_feats = new_feats[hard_inds]
        self.neg_cache_feats.append(new_feats)

    def update_model(self, new_neg_feats, force=False):
        # update cache
        if new_neg_feats is not None:
            self._dilate_neg_cache(new_neg_feats)

        # train model
        cache_size = sum([f.shape[0] for f in self.neg_cache_feats])
        if cache_size >= self.limit_retrain or force:
            # train and clean cache
            self._shrink_neg()
            self.neg_feats = np.vstack([self.neg_feats, *self.neg_cache_feats])
            self.neg_cache_feats = []
            self._train()

    def parameters(self):
        return np.hstack((self.svm_object.intercept_.ravel(), self.svm_object.coef_.ravel()))

    def score(self, feats):
        return self.svm_object.decision_function(feats)

    def calibrate(self):
        if self.initialized:
            logger.info("Training calibrated SVM model {} for {} positive samples and {} negative samples..."
                        .format(self.name, self.pos_feats.shape[0], self.neg_feats.shape[0]))
            X = np.vstack([self.pos_feats, self.neg_feats])
            Y = np.vstack([np.ones((self.pos_feats.shape[0], 1)), -1.0 * np.ones((self.neg_feats.shape[0], 1))])
            self.clb_object.fit(X, Y.ravel())
            acc = self.clb_object.score(X, Y.ravel())
            logger.info("Calibrated SVM {} Trained. Accuracy on training {}.".format(self.name, acc))
