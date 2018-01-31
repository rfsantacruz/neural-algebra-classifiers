import abc, logging, os, anytree
from dbac_lib.dbac_primitives import IPrimitiveCollection, HardMiningSVM
from dbac_lib.dbac_feature_ext import IFeatureExtractor
from dbac_lib.dbac_util import CycleIterator, batch_iterator, TicToc
from dbac_lib import dbac_expression
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
from joblib import Parallel, delayed
from sklearn import svm as sklearn_svm
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
from dbac_lib import vgg_preprocessing

_logger = logging.getLogger(__name__)
_MODEL_NAMES = ['cha', 'sup', 'ind', 'nba_mlp']


def exp_members(exp_tup):
    op, v_a, v_b = exp_tup[0], int(exp_tup[1]), int(exp_tup[2]) if exp_tup[2] != 'None' else None
    return op, v_a, v_b


class IModel(metaclass=abc.ABCMeta):
    def __init__(self, name, feat_ext, prim_rpr):
        assert name in _MODEL_NAMES
        assert isinstance(feat_ext, IFeatureExtractor)
        assert isinstance(prim_rpr, IPrimitiveCollection)
        self.name = name
        self.feat_ext = feat_ext
        self.prim_rpr = prim_rpr

    @staticmethod
    def factory(name, feat_ext, prim_rpr, **kwargs):
        if name == _MODEL_NAMES[0]:
            return Chance(feat_ext, prim_rpr, **kwargs)
        elif name == _MODEL_NAMES[1]:
            return Supervised(feat_ext, prim_rpr, **kwargs)
        elif name == _MODEL_NAMES[2]:
            return Independent(feat_ext, prim_rpr, **kwargs)
        elif name == _MODEL_NAMES[3]:
            return NBA_MLP(feat_ext, prim_rpr, **kwargs)
        else:
            raise ValueError("Model {} is not defined.".format(name))

    @abc.abstractmethod
    def learning(self, images_path, labels, expressions, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def inference(self, expressions, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def score(self, images_path, expressions, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, file_path):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, file_path):
        raise NotImplementedError()


class Chance(IModel):
    def __init__(self, feat_ext, prim_rpr, **kwargs):
        super().__init__(_MODEL_NAMES[0], feat_ext, prim_rpr)

    def inference(self, expressions, **kwargs):
        _logger.info("Chance model cannot infer classifiers.")

    def learning(self, images_path, labels, expressions, **kwargs):
        _logger.info("Chance model does not require learning.")

    def score(self, images_path, expressions, **kwargs):
        return np.random.rand(len(expressions), images_path.shape[0])

    def save(self, file_path):
        _logger.info("Chance model does not require save.")

    def load(self, file_path):
        _logger.info("Chance model does not require load.")


class Supervised(IModel):
    def __init__(self, feat_ext, prim_rpr, **kwargs):
        super().__init__(_MODEL_NAMES[1], feat_ext, prim_rpr)
        self.cls_dic = dict()

    def inference(self, expressions, **kwargs):
        dims = self.cls_dic.get(list(self.cls_dic.keys())[0]).shape
        ret = [self.cls_dic.get(self._exp2key(exp), np.random.rand(*dims)) for exp in expressions]
        return ret

    def learning(self, images_path, labels, expressions, **kwargs):
        num_threads = int(kwargs.get('num_threads', 10))
        image_feats = self.feat_ext.compute(images_path)
        svm_parameters = Parallel(n_jobs=num_threads)(
            delayed(_compute_svm_params)(image_feats, labels, expressions, exp_idx) for exp_idx in
            range(len(expressions)))
        for idx, (exp_lst, param) in enumerate(zip(expressions, svm_parameters)):
            self.cls_dic[self._exp2key(exp_lst)] = param

    def score(self, images_path, expressions, **kwargs):
        images_feat = self.feat_ext.compute(images_path)
        expressions_w = np.vstack(self.inference(expressions))
        scs = np.reshape(expressions_w[:, 0], (-1, 1)) + np.dot(expressions_w[:, 1:], images_feat.T)
        return scs

    def save(self, file_path):
        np.save(file_path, self.cls_dic)

    def load(self, file_path):
        self.cls_dic = np.load(file_path).item()

    def _exp2key(self, exp):
        if isinstance(exp, anytree.node.Node):
            return dbac_expression.exp2list_parse(exp)
        elif isinstance(exp, (list, tuple, np.ndarray)):
            return dbac_expression.exp2list_parse(dbac_expression.list2exp_parse(exp))
        else:
            raise ValueError("Not supported expression format")


def _compute_svm_params(img_feats, prim_labels, expressions, exp_idx):
    # setup svm
    exp_lst = expressions[exp_idx]
    _logger.info("{}/{} - Training svm  ...".format(exp_idx, len(expressions)))
    exp_tree = dbac_expression.list2exp_parse(exp_lst)
    var_dic = {p: prim_labels[:, int(p)] for p in dbac_expression.get_vars(exp_tree)}
    exp_labels = dbac_expression.eval_exp(exp_tree, var_dic)
    svm_object = sklearn_svm.LinearSVC(C=1e-5, class_weight={1: 2.0, 0: 1.0}, verbose=0, penalty='l2',
                                       loss='hinge', dual=True)
    svm_object.fit(img_feats, exp_labels)
    train_acc = svm_object.score(img_feats, exp_labels)
    _logger.info("{}/{} - Finalized svm. Positives {}, Negatives {}, Accuracy {}."
                 .format(exp_idx, len(expressions), np.sum(exp_labels), np.sum(np.logical_not(exp_labels)), train_acc))
    svm_params = np.hstack((svm_object.intercept_.ravel(), svm_object.coef_.ravel()))
    return svm_params


class Independent(IModel):
    def __init__(self, feat_ext, prim_rpr, **kwargs):
        super().__init__(_MODEL_NAMES[2], feat_ext, prim_rpr)

    def learning(self, images_path, labels, expressions, **kwargs):
        _logger.info("Independent model does not require learning.")

    def inference(self, expressions, **kwargs):
        _logger.info("Independent model cannot infer classifiers.")

    def score(self, images_path, expressions, **kwargs):
        scores = np.zeros((len(expressions), images_path.shape[0]), dtype=np.float)
        images_feat = self.feat_ext.compute(images_path)
        ops_dic = {dbac_expression.OPS[0]: lambda v: 1.0 - v, dbac_expression.OPS[1]: np.multiply,
                   dbac_expression.OPS[2]: lambda v1, v2: (v1 + v2) - np.multiply(v1, v2)}
        var_dic = {str(p): self.prim_rpr.get_cls(int(p))[0].predict_proba(images_feat)[:, 1] for p in
                   self.prim_rpr.get_ids()}
        for idx, exp_lst in enumerate(expressions):
            exp_tree = dbac_expression.list2exp_parse(exp_lst)
            scores[idx] = dbac_expression.eval_exp(exp_tree, var_dic, ops_dic)
            if idx % 100 == 0:
                _logger.info("Tested for {}/{} expressions.".format(idx, len(expressions)))
        return scores

    def save(self, file_path):
        _logger.info("Indepedent model does not require save.")

    def load(self, file_path):
        _logger.info("Indepedent model does not require load.")


class NBA_MLP(IModel):
    def __init__(self, feat_ext, prim_rpr, **kwargs):
        super().__init__(_MODEL_NAMES[3], feat_ext, prim_rpr)
        # tensorflow graph and session
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.log_device_placement = True
        config.allow_soft_placement = True
        self.tf_session = tf.Session(graph=self.graph, config=config)
        self.dim = self.prim_rpr.get_rpr(self.prim_rpr.get_ids()[0]).shape[-1]
        self.learn_feats = bool(int(kwargs.get('learn_feats', False)))
        self.is_train = bool(int(kwargs.get('is_train', True)))
        self.norm_in = bool(int(kwargs.get('norm_in', False)))
        self.norm_out = bool(int(kwargs.get('norm_out', False)))
        self.demorgan_reg = bool(int(kwargs.get('demorgan_reg', False)))
        self.net_type = int(kwargs.get('net_type', 0))  # 0: NOT AND OR, 1: NOT AND, 2: NOT OR
        _logger.info("Model parameters: learn_feats={}, is_train={}, norm_in={}, norm_out={}, demorgan_reg={}, "
                     "net_type={}".format(self.learn_feats, self.is_train, self.norm_in, self.norm_out,
                                          self.demorgan_reg, self.net_type))

        # Model definition
        with self.graph.as_default() as graph:
            # input tensors
            self._prims_rpr_ph = tf.placeholder(tf.float32, (None, 2 * self.dim))
            self._ground_truth_ph = tf.placeholder(tf.float32, (None, None))
            self._switch_ph = tf.placeholder(tf.int32, (None,))
            self.is_training_ph = tf.placeholder(tf.bool)

            # normalize inputs
            if self.norm_in:
                prims_rpr_a, prims_rpr_b = tf.split(self._prims_rpr_ph, num_or_size_splits=2, axis=1)
                prims_rpr_a, prims_rpr_b = tf.nn.l2_normalize(prims_rpr_a, dim=1), tf.nn.l2_normalize(prims_rpr_b, dim=1)
                prims_rpr_tn = tf.concat([prims_rpr_a, prims_rpr_b], axis=1)
            else:
                prims_rpr_tn = 1.0 * self._prims_rpr_ph

            # mlps and multiplexing
            not_ex = lambda input_tn: -1.0 * tf.slice(input_tn, [0, 0],
                                                      [tf.shape(input_tn)[0], self.dim])
            and_mlp = lambda input_tn, reuse=False: _mlp(input_tn, [int(np.ceil(1.5 * self.dim)), self.dim],
                                                         scope='nba_mlp/AND', reuse=reuse)
            or_mlp = lambda input_tn, reuse=False: _mlp(input_tn, [int(np.ceil(1.5 * self.dim)), self.dim],
                                                        scope='nba_mlp/OR', reuse=reuse)
            zero_tn = tf.zeros((tf.shape(prims_rpr_tn)[0], self.dim), tf.float32)

            if self.net_type == 0:
                # NOT AND OR
                self.output_tn = tf.where(tf.equal(self._switch_ph, 0), not_ex(prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 1), and_mlp(prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 2), or_mlp(prims_rpr_tn), zero_tn)
            elif self.net_type == 1:
                # NOT AND
                self.output_tn = tf.where(tf.equal(self._switch_ph, 0), not_ex(prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 1), and_mlp(prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 2), -1.0 * and_mlp(-1.0 * prims_rpr_tn, reuse=True), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 50), or_mlp(prims_rpr_tn), zero_tn)
            elif self.net_type == 2:
                # NOT OR
                self.output_tn = tf.where(tf.equal(self._switch_ph, 0), not_ex(prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 1), -1.0 * or_mlp(-1.0 * prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 2), or_mlp(prims_rpr_tn, reuse=True), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 50), and_mlp(prims_rpr_tn), zero_tn)
            else:
                raise ValueError("Network Type is not supported! net_type={}".format(self.net_type))

            # Regularization based on De Morgan laws
            if self.demorgan_reg:
                self.dm_output_tn = -1.0 * prims_rpr_tn
                self.dm_output_tn = tf.where(tf.equal(self._switch_ph, 0), not_ex(prims_rpr_tn), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 1), or_mlp(prims_rpr_tn, True), zero_tn) \
                                 + tf.where(tf.equal(self._switch_ph, 2), and_mlp(prims_rpr_tn, True), zero_tn)
                self.dm_output_tn = -1.0 * self.dm_output_tn

            # normalize output
            if self.norm_out:
                self.output_tn = tf.nn.l2_normalize(self.output_tn, dim=1)
                if self.demorgan_reg:
                    self.dm_output_tn = tf.nn.l2_normalize(self.dm_output_tn, dim=1)

            # visual branch
            if self.learn_feats:
                with tf.device('/gpu:0'):
                    self._images_ph = tf.placeholder(tf.string, (None,))
                    img_prep_func = lambda img_path: vgg_preprocessing.preprocess_image(
                        tf.image.decode_jpeg(tf.read_file(img_path), channels=3), 224, 224, is_training=self.is_train)
                    processed_images = tf.map_fn(img_prep_func, self._images_ph, dtype=tf.float32)
                    with slim.arg_scope(vgg.vgg_arg_scope()):
                        logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=self.is_training_ph)
                        self._images_feats = tf.squeeze(graph.get_tensor_by_name('vgg_16/fc6/BiasAdd:0'))
                        self.init_vgg = slim.assign_from_checkpoint_fn('/data/home/rfsc/dbac/models/vgg_16.ckpt',
                                                                       slim.get_model_variables('vgg_16'))
                self.tf_saver_feats = tf.train.Saver(tf_slim.get_model_variables(scope='vgg_16'))
            else:
                self._images_ph = tf.placeholder(tf.float32, (None, self.dim - 1))
                self._images_feats = 1.0 * self._images_ph
                self.init_vgg = None

            # score images
            self.scores_tn = tf.add(
                tf.reshape(self.output_tn[:, 0], (-1, 1)),
                tf.matmul(self.output_tn[:, 1:], self._images_feats, transpose_b=True))
            # Checkpoint save and restore
            self.tf_saver_and = tf.train.Saver(tf_slim.get_model_variables(scope='nba_mlp/AND'))
            self.tf_saver_or = tf.train.Saver(tf_slim.get_model_variables(scope='nba_mlp/OR'))

    def learning(self, images_path, labels, expressions, **kwargs):

        # training parameters
        batch_size, num_epochs = int(kwargs.get('batch_size', 32)), int(kwargs.get('num_epochs', 1e3))
        snap_int, snap_dir, log_dir = int(kwargs.get('snap_int', 250)), kwargs.get('snap_dir', None), kwargs.get(
            'log_dir', None)
        init_weights = kwargs.get('init_weights', None)
        snapshot = kwargs.get('snapshot', None)
        learning_rate = float(kwargs.get('learning_rate', 1e-5))
        alphas = [float(p) for p in kwargs.get('alphas', '10.0 1.0 0.1 1.0').split()]
        _logger.info("Training parameters: batch_size={}, num_epochs={}, snap_int={}, snap_dir={}, log_dir={}, "
                     "learning_rate={}, alphas={}, learn_feats={}, init_weights={}, norm_in={}, norm_out={},"
                     " snapshot={}, demorgan_reg={}".format(batch_size, num_epochs, snap_int, snap_dir, log_dir, learning_rate, alphas,
                                                            self.learn_feats, init_weights, self.norm_in, self.norm_out,
                                                            snapshot, self.demorgan_reg))

        # setup training network
        with self.graph.as_default() as graph:
            # Loss
            reg_loss = tf.reduce_mean(tf.losses.get_regularization_loss())
            norm_loss = tf.reduce_mean(0.5 * tf.pow(tf.norm(self.output_tn, axis=-1), 2.0))
            cls_loss = tf.losses.hinge_loss(self._ground_truth_ph, self.scores_tn, reduction=tf.losses.Reduction.MEAN)
            loss_tn = alphas[0] * norm_loss + alphas[1] * cls_loss + alphas[2] * reg_loss
            if self.demorgan_reg:
                dem_loss = tf.reduce_mean(0.5 * tf.pow(tf.norm(self.output_tn - self.dm_output_tn, axis=-1), 2.0))
                loss_tn = loss_tn + (alphas[3] * dem_loss)
                dem_loss_val, dem_loss_up, dem_loss_reset = _create_reset_metric(
                    tf.metrics.mean, 'epoch_dem_loss', values=dem_loss)
                tf.summary.scalar('dem_loss', dem_loss_val)

            pred = tf.greater(self.scores_tn, 0.0)
            # Metrics
            reg_loss_val, reg_loss_up, reg_loss_reset = _create_reset_metric(
                tf.metrics.mean, 'epoch_reg_loss', values=reg_loss)
            tf.summary.scalar('reg_loss', reg_loss_val)
            cls_loss_val, cls_loss_up, cls_loss_reset = _create_reset_metric(
                tf.metrics.mean, 'epoch_cls_loss', values=cls_loss)
            tf.summary.scalar('cls_loss', cls_loss_val)
            norm_loss_val, norm_loss_up, norm_loss_reset = _create_reset_metric(
                tf.metrics.mean, 'epoch_norm_loss', values=norm_loss)
            tf.summary.scalar('norm_loss', norm_loss_val)
            loss_val, loss_up, loss_reset = _create_reset_metric(
                tf.metrics.mean, 'epoch_loss', values=loss_tn)
            tf.summary.scalar('total_loss', loss_val)
            prec_val, prec_up, prec_reset = _create_reset_metric(
                tf.metrics.precision, 'epoch_prec', predictions=pred, labels=self._ground_truth_ph)
            tf.summary.scalar('Precision', prec_val)
            rec_val, rec_up, rec_reset = _create_reset_metric(
                tf.metrics.recall, 'epoch_rec', predictions=pred, labels=self._ground_truth_ph)
            tf.summary.scalar('Recall', rec_val)
            tf.summary.scalar('Fscore', (2 * prec_val * rec_val) / (prec_val + rec_val + 1e-6))
            summ_ops = tf.summary.merge_all()
            summ_writer = tf.summary.FileWriter(log_dir) if log_dir else None
            metrics_ops_reset = [reg_loss_reset, cls_loss_reset, norm_loss_reset, loss_reset, prec_reset, rec_reset]
            metrics_ops_update = [reg_loss_up, cls_loss_up, norm_loss_up, loss_up, prec_up, rec_up]
            if self.demorgan_reg:
                metrics_ops_reset += [dem_loss_reset]
                metrics_ops_update += [dem_loss_up]
            # Optimizer
            global_step_tn = tf.train.get_or_create_global_step(graph)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss_tn, global_step=global_step_tn, colocate_gradients_with_ops=True)
            init = tf.global_variables_initializer()
            tf_snap_tr = tf.train.Saver(max_to_keep=1)

        # Decompose expressions and compute labels
        _logger.info("Decomposing expressions...")
        valid_exps, pos_img_ites, neg_img_ites, exp_labels = [], [], [], []
        for exp_lst in expressions:
            for exp_term in dbac_expression.get_terms(dbac_expression.list2exp_parse(exp_lst)):
                term_lst = dbac_expression.exp2list_parse(exp_term)
                var_dic = {p: labels[:, int(p)] for p in dbac_expression.get_vars(exp_term)}
                term_labels = dbac_expression.eval_exp(exp_term, var_dic)
                if (term_lst not in valid_exps) and (exp_term.name != dbac_expression.OPS[0]) \
                        and (term_labels.sum() > 0) and (np.logical_not(term_labels).sum() > 0):
                    valid_exps.append(term_lst)
                    exp_labels.append(term_labels)
                    pos_img_ites.append(CycleIterator(list(np.where(term_labels)[0])))
                    neg_img_ites.append(CycleIterator(list(np.where(np.logical_not(term_labels))[0])))
        expressions = valid_exps
        exp_ite = CycleIterator(np.arange(len(expressions)).tolist())
        exp_labels = np.vstack(exp_labels).astype(np.float32)
        _logger.info("Total of expressions decomposed: {}".format(len(expressions)))

        # Initialization
        _logger.info("Initializing model...")
        self.tf_session.run(init)
        if self.init_vgg is not None:
            _logger.info("Loading features pre-trained weights")
            self.init_vgg(self.tf_session)
        if init_weights:
            _logger.info("Loading model pre-trained weights")
            self.load(init_weights)
        init_epoch = 0
        if snapshot:
            _logger.info("Loading from training snapshot")
            tf_snap_tr.restore(self.tf_session, snapshot)
            init_epoch = int((self.tf_session.run(global_step_tn) * batch_size) / len(expressions))

        # training loop
        _logger.info("Training...")
        for epoch in range(init_epoch, num_epochs):
            self.tf_session.run(metrics_ops_reset)
            for b in range(int(np.ceil(len(expressions) / batch_size))):
                # batch sampling
                b_exp_ids = [next(exp_ite) for _ in range(batch_size)]
                b_img_ids = [next(pos_img_ites[exp_id]) for _ in range(5) for exp_id in b_exp_ids]
                b_img_ids += [next(neg_img_ites[exp_id]) for _ in range(5) for exp_id in b_exp_ids]

                # compute image features
                if self.learn_feats:
                    b_img_feats = images_path[b_img_ids]
                else:
                    b_img_feats = self.feat_ext.compute(images_path[b_img_ids])

                # compute operations
                b_prims_rpr, b_op_switch = [], []
                for exp_id in b_exp_ids:
                    exp_tree = dbac_expression.list2exp_parse(expressions[exp_id])
                    b_op_switch.append({'NOT': 0, 'AND': 1, 'OR': 2}[exp_tree.name])
                    operand_a, operand_b = exp_tree.children if np.random.rand() > 0.5 else exp_tree.children[::-1]
                    b_prims_rpr.append(dbac_expression.exp2list_parse(operand_a))
                    b_prims_rpr.append(dbac_expression.exp2list_parse(operand_b))
                b_op_switch = np.array(b_op_switch)
                b_prims_rpr = self.inference(b_prims_rpr).reshape((len(b_exp_ids), 2 * self.dim))

                # compute labels
                b_exp_labels = exp_labels[b_exp_ids, :]
                b_exp_labels = b_exp_labels[:, b_img_ids]

                # run model
                self.tf_session.run(
                    [train_op, loss_tn] + metrics_ops_update,
                    feed_dict={self.is_training_ph: True,
                               self._images_ph: b_img_feats,
                               self._prims_rpr_ph: b_prims_rpr,
                               self._switch_ph: b_op_switch,
                               self._ground_truth_ph: b_exp_labels})

            if (epoch + 1) % 2 == 0:
                loss, prec, rec, summary = self.tf_session.run([loss_val, prec_val, rec_val, summ_ops])
                _logger.info("Epoch {}: Loss={:.4f}, Prec={:.2f}, Rec={:.2f}, Fsc={:.2f}"
                             .format((epoch + 1), loss, prec, rec, (2 * prec * rec) / (prec + rec + 1e-6)))
                if summ_writer:
                    summ_writer.add_summary(summary, global_step=epoch + 1)

            if snap_dir and (epoch + 1) % snap_int == 0:
                snap_file = os.path.join(snap_dir, 'nba_mlp_snap_E{}.npz'.format((epoch + 1)))
                self.save(snap_file)
                tf_snap_tr.save(self.tf_session, os.path.join(snap_dir, 'train.chk'), latest_filename='checkpoint.TRAIN')
                _logger.info("Model epoch {} snapshoted to {}".format(epoch + 1, snap_file))

    def score(self, images_path, expressions, **kwargs):
        images_feat = []
        _logger.info("Computing image representation")
        for b_img_paths in batch_iterator(10, images_path):
            b_img_paths = list(filter(lambda f: f is not None, b_img_paths))
            if b_img_paths:
                if self.learn_feats:
                    feats = self.tf_session.run(self._images_feats, feed_dict={self.is_training_ph: False,
                                                                               self._images_ph: b_img_paths})
                else:
                    feats = self.feat_ext.compute(b_img_paths)
                images_feat.append(feats)
        images_feat = np.vstack(images_feat)
        _logger.info("Computing expression classifiers")
        exp_rpr = self.inference(expressions)
        b, A = np.reshape(exp_rpr[:, 0], (-1, 1)), exp_rpr[:, 1:]
        _logger.info("Computing scores")
        scores = np.dot(A, images_feat.T) + b
        return scores

    def inference(self, expressions, **kwargs):
        ops_dic = dict()
        ops_dic[dbac_expression.OPS[0]] = lambda v1: -1 * v1
        ops_dic[dbac_expression.OPS[1]] = lambda v1, v2: np.ravel(self.tf_session.run(
            self.output_tn, feed_dict={self._prims_rpr_ph: np.expand_dims(np.hstack([v1, v2]), 0),
                                       self._switch_ph: 1 * np.ones(1, dtype=np.int)}))
        ops_dic[dbac_expression.OPS[2]] = lambda v1, v2: np.ravel(self.tf_session.run(
            self.output_tn, feed_dict={self._prims_rpr_ph: np.expand_dims(np.hstack([v1, v2]), 0),
                                       self._switch_ph: 2 * np.ones(1, dtype=np.int)}))
        var_dic = {str(p): self.prim_rpr.get_rpr(int(p))[0] for p in self.prim_rpr.get_ids()}
        exp_rpr = np.zeros((len(expressions), self.dim), dtype=np.float)
        for idx, exp_lst in enumerate(expressions):
            exp_tree = dbac_expression.list2exp_parse(exp_lst)
            exp_rpr[idx] = dbac_expression.eval_exp(exp_tree, var_dic, ops_dic)
            if (idx+1) % 250 == 0:
                _logger.info("Inference expression classifiers {}/{}.".format(idx, len(expressions)))
        return exp_rpr

    def load(self, file_path):
        self.tf_saver_and.restore(self.tf_session, "{}.AND.chk".format(file_path))
        self.tf_saver_or.restore(self.tf_session, "{}.OR.chk".format(file_path))
        feat_var_file = "{}.FEAT.chk".format(file_path)
        if self.learn_feats and tf.gfile.Glob("{}*".format(feat_var_file)):
            self.tf_saver_feats.restore(self.tf_session, feat_var_file)

    def save(self, file_path):
        self.tf_saver_and.save(self.tf_session, "{}.AND.chk".format(file_path), latest_filename='checkpoint.AND')
        self.tf_saver_or.save(self.tf_session, "{}.OR.chk".format(file_path), latest_filename='checkpoint.OR')
        if self.learn_feats:
            self.tf_saver_feats.save(self.tf_session, "{}.FEAT.chk".format(file_path), latest_filename='checkpoint.FEAT')


def _leaky_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)


def _create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def _mlp(input_tn, dims, l2reg=0.001, scope='mlp', reuse=False):
    x = tf_slim.fully_connected(input_tn, dims[0], scope='{}/fc1'.format(scope), reuse=reuse, activation_fn=_leaky_relu,
                                weights_regularizer=tf_slim.l2_regularizer(l2reg))
    x = tf_slim.fully_connected(x, dims[1], scope='{}/fc2'.format(scope), reuse=reuse, activation_fn=None,
                                weights_regularizer=tf_slim.l2_regularizer(l2reg))
    return x
