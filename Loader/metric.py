"""Pascal VOC Classification evaluation."""
from __future__ import division
import numpy as np
import torch
import math
import pdb



class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.

        preds : list of NDArray
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)


    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()


    
    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0


    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)


    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))



class VOCMultiClsMApMetric(EvalMetric):
    """
    Calculate mean AP for MultiClassification task

    Parameters:
    ---------
    class_names : list of str
        Required, list of class_name
    """
    def __init__(self, class_names=None, ignore_label=None, voc_action_type=False):
        assert isinstance(class_names, (list, tuple))
        for name in class_names:
            assert isinstance(name, str), "must provide names as str"
        self.class_names = class_names
        super(VOCMultiClsMApMetric, self).__init__('VOCMeanAP')
        num = len(class_names)
        self.name = list(class_names) + ['mAP']
        self.class_names = class_names
        self.num = num + 1
        self.ignore_label = ignore_label
        self.voc_action_type = voc_action_type
        self._scores = []
        self._labels = []
        self.reset()

    def reset(self):
        """Clear the internal statistics to initial state."""
        self._scores = []
        self._labels = []
        for i in range(len(self.class_names)):
            self._scores.append([])
            self._labels.append([])

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : numpy.float32
           Value of the evaluation.
        """
        names = ['%s' % (self.name[i]) for i in range(self.num)]
        values = self._update()
        if self.voc_action_type:
            names.append('mAP(Without Other)')
            values = np.append(values, np.mean(values[:-2]))
        return names, values

    def save(self, file_name='result.csv'):
        """Save scores and labels."""
        labels = np.array(self._labels)
        scores = np.array(self._scores)
        res = np.concatenate((labels, scores), axis=0).transpose()
        np.savetxt(file_name, res, fmt='%.6f', delimiter=",")

    def load(self, file_name):
        res = np.loadtxt(file_name, delimiter=',')
        self.load_from_nparray(res[:, 0:len(self.class_names)], res[:, len(self.class_names):])

    def load_from_nparray(self, labels, scores):
        assert scores.shape[1] == len(self.class_names), \
            'Scores must have the shape(N, %d).' % len(self.class_names)
        assert labels.shape[1] == len(self.class_names), \
            'Labels must have the shape(N, %d).' % len(self.class_names)
        self._labels = np.asarray(labels.transpose())
        self._scores = np.asarray(scores.transpose())

    def update(self, pred_scores, gt_labels):
        """Update internal buffer with latest prediction and gt pairs.

        Parameters
        ----------
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction class scores with shape `B, N, C`.
        gt_labels : mxnet.NDArray or numpy.ndarray
            Ground-truth labels with shape `B, N`.
        """
        #pdb.set_trace()
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x.numpy() if isinstance(x, type(torch.tensor([]))) else x for x in a]
                #print(out)
                out = np.array(out)
                #print(out)
                # just return out directly for 1-d array
                if len(out.shape) == 1:
                    return out
                return np.concatenate(out, axis=0)
            elif isinstance(a, type(torch.tensor([]))):
                a = a.numpy()
            return a

        num_class = len(self.class_names)
        # Split in batch axis
        for pred_score, gt_label in zip(*[as_numpy(x.to("cpu")) for x in [pred_scores, gt_labels]]):
            # pred_score shape (N, C), gt_label shape (N, C)
            pred_score = pred_score.reshape((-1, num_class))
            gt_label = gt_label.reshape((-1, num_class))
            assert pred_score.shape[0] == gt_label.shape[0], 'Num of scores must be the same with num of ground truths.'
            gt_label = gt_label.astype(int)
            #print('gt_label:',gt_label)
            #print('pred_scores:',pred_scores)
            # Iterate over classes
            for i in range(len(self.class_names)):
                single_class_score = pred_score[:, i]
                single_class_label = gt_label[:, i]
                self._scores[i].extend(single_class_score.tolist())
                self._labels[i].extend(single_class_label.tolist())

    def _update(self):
        #pdb.set_trace()
        """ update num_inst and sum_metric """
        ap_list = np.zeros(self.num, dtype=np.float32)
        labels = np.array(self._labels)
        scores = np.array(self._scores)

        for a in range(len(self.class_names)):
            single_class_label = labels[a]
            single_class_score = scores[a]
            if self.ignore_label is not None:
                valid_index = np.where(single_class_label != self.ignore_label)
                single_class_score = single_class_score[valid_index]
                single_class_label = single_class_label[valid_index]
            tp = single_class_label == 1
            npos = np.sum(tp, axis=0)
            fp = single_class_label != 1
            sc = single_class_score
            cat_all = np.vstack((tp, fp, sc)).transpose()
            ind = np.argsort(cat_all[:, 2])
            cat_all = cat_all[ind[::-1], :]
            tp = np.cumsum(cat_all[:, 0], axis=0)
            fp = np.cumsum(cat_all[:, 1], axis=0)

            # # Compute precision/recall
            if (npos==0):
                rec = tp / (npos + 0.001)
            else:
                rec = tp / npos
            prec = np.divide(tp, (fp + tp))
            ap_list[a] = self._average_precision(rec, prec)
        ap_list[-1] = np.mean(np.nan_to_num(ap_list[:-1]))
        return ap_list

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        rec = rec.reshape(rec.size, 1)
        prec = prec.reshape(prec.size, 1)
        z = np.zeros((1, 1))
        o = np.ones((1, 1))
        mrec = np.vstack((z, rec, o))
        mpre = np.vstack((z, prec, z))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # i = find(mrec(2:end)~=mrec(1:end-1))+1;
        I = np.where(mrec[1:] != mrec[0:-1])[0] + 1
        ap = 0
        for i in I:
            ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap


class RCNNAccMetric(EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        # pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        pred_label = np.argmax(rcnn_cls,axis=-1)
        # num_acc = mx.nd.sum(pred_label == rcnn_label)
        num_acc = np.sum(pred_label==rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class CosineAnnealingSchedule:
    def __init__(self, min_lr, max_lr, cycle_length):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr


class Loss(EvalMetric):
    """Dummy metric for directly printing loss.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name='loss',
                 output_names=None, label_names=None):
        super(Loss, self).__init__(
            name, output_names=output_names, label_names=label_names,
            has_global_stats=True)

    def update(self, _, preds):

        if isinstance(preds, np.array):
            preds = [preds]

        for pred in preds:
            loss = np.sum(pred).asscalar()
            self.sum_metric += loss
            self.global_sum_metric += loss
            self.num_inst += pred.size
            self.global_num_inst += pred.size