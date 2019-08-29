"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import warnings

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SyncBatchNorm

from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from ..rcnn import RCNN
from ..rpn import RPN
from ...nn.feature import FPNFeatureExpander

__all__ = ['FasterRCNN', 'get_faster_rcnn',
           'faster_rcnn_resnet50_v1b_voc',
           'faster_rcnn_resnet50_v1b_coco',
           'faster_rcnn_fpn_resnet50_v1b_coco',
           'faster_rcnn_fpn_bn_resnet50_v1b_coco',
           'faster_rcnn_resnet50_v1b_custom',
           'faster_rcnn_resnet101_v1d_voc',
           'faster_rcnn_resnet101_v1d_coco',
           'faster_rcnn_fpn_resnet101_v1d_coco',
           'faster_rcnn_resnet101_v1d_custom']


class FasterRCNN(RCNN):
    r"""Faster RCNN network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    box_features : gluon.HybridBlock, default is None
        feature head for transforming shared ROI output (top_features) for box prediction.
        If set to None, global average pooling will be used.
    short : int, default is 600.
        Input image short side size.
    max_size : int, default is 1000.
        Maximum size of input image long side.
    min_stage : int, default is 4
        Minimum stage NO. for FPN stages.
    max_stage : int, default is 4
        Maximum stage NO. for FPN stages.
    train_patterns : str, default is None.
        Matching pattern for trainable parameters.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str, default is align
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2, default is (14, 14)
        (height, width) of the ROI region.
    strides : int/tuple of ints, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
        For FPN, use a tuple of ints.
    clip : float, default is None
        Clip bounding box target to this value.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float, default is (8, 16, 32)
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float, default is (0.5, 1, 2)
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    rpn_train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training of RPN.
    rpn_train_post_nms : int, default is 2000
        Return top proposal results after NMS in training of RPN.
        Will be set to rpn_train_pre_nms if it is larger than rpn_train_pre_nms.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
        Will be set to rpn_test_pre_nms if it is larger than rpn_test_pre_nms.
    rpn_nms_thresh : float, default is 0.7
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training.
    train_post_nms : int, default is 2000
        Return top proposal results after NMS in training.
    test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing.
    test_post_nms : int, default is 300
        Return top proposal results after NMS in testing.
    rpn_min_size : int, default is 16
        Proposals whose size is smaller than ``min_size`` will be discarded.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    max_num_gt : int, default is 300
        Maximum ground-truth number in whole training dataset. This is only an upper bound, not
        necessarily very precise. However, using a very big number may impact the training speed.
    additional_output : boolean, default is False
        ``additional_output`` is only used for Mask R-CNN to get internal outputs.
    force_nms : bool, default is False
        Appy NMS to all categories, this is to avoid overlapping detection results from different
        categories.

    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    force_nms : bool
        Appy NMS to all categories, this is to avoid overlapping detection results
        from different categories.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    target_generator : gluon.Block
        Generate training targets with boxes, samples, matches, gt_label and gt_box.

    """
    '''
    这里的注释以faster_rcnn_fpn_resnet50_v1b_coco为例
    features----FPN中多尺度输出的不同大小的特征图,[P2, P3, P4, P5]
    top_features----Tail feature extractor after feature pooling layer(暂时没有做出很好的对应),None
    classes---特征的种类
    box_features----2 FC layer before RCNN cls and reg,这里是两个1024的FC layer
    short, max_size----图片送入进来时,resize的长短边界限
    min_stage, max_stage----这里应该指的是FPN输出的阶数中最小和最大的序号
    train_patterns----(TO_DO)暂时还没有理解这个参数的作用
    nms_thresh----nms中的iou阈值, 0.5
    nms_topk----对预测得分前k个进行nms操作, -1
    post_nms----nms后,输出前多少的结果, -1
    roi_mode----选择roi-pooling或者roi-align
    roi_size----roi-region的大小, (7,7)
    strides----(4, 8, 16, 32, 64), 这里其实是输出了4个stage,64这一层是怎么来的(TO_DO)
    clip---- Clip bounding box target to this value, 4.14(TO_DO)没有明白这里为什么要设置成4.14
    rpn_channel----Channel number used in RPN convolutional layers, 1024
    base_size, scales, ratios, alloc_size----设置anchor时会使用得到的参数
    rpn_nms_thresh----训练过程中rpn中nms处理过程中的iou阈值
    rpn_train_pre_nms, rpn_train_post_nms----train过程中,nms的一些参数
    rpn_test_pre_nms, rpn_test_post_nms----test过程中, nms的一些参数
    rpn_min_size----rpn最后输出框的最小尺寸,当nms之后需要输出的框的尺寸小于这个尺寸时,
    直接取消掉, 1
    num_sample----作为RCNN过程的目标框数量, 512
    pos_iou_thresh----RCNN的target生成过程中会有使用到,0.5,指输出框与目标框的iou高于0.5即认为是
    正类框
    pos_ratio----正样本采样的比例,这里就是指,经过上面匹配出来的正样本并不是每一个都会用于RCNN的
    训练,而是会按照一定的比例进行采样处理
    additional_output----在mask
    '''
    def __init__(self, features, top_features, classes, box_features=None,
                 short=600, max_size=1000, min_stage=4, max_stage=4, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100,
                 roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
                 rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=300,
                 additional_output=False, force_nms=False, **kwargs):
        # 这里一定要注意
        # 这里就是利用super方法,调用了RCNN的初始化方法,也就是相当于初始化生成了一个RCNN模块
        super(FasterRCNN, self).__init__(
            features=features, top_features=top_features, classes=classes,
            box_features=box_features, short=short, max_size=max_size,
            train_patterns=train_patterns, nms_thresh=nms_thresh, nms_topk=nms_topk,
            post_nms=post_nms, roi_mode=roi_mode, roi_size=roi_size, strides=strides, clip=clip,
            force_nms=force_nms, **kwargs)
        if rpn_train_post_nms > rpn_train_pre_nms:
            rpn_train_post_nms = rpn_train_pre_nms
        if rpn_test_post_nms > rpn_test_pre_nms:
            rpn_test_post_nms = rpn_test_pre_nms

        self.ashape = alloc_size[0]
        # 这里根据输入参数来确定,在选定的示例中,min_stage = 2, max_stage = 6
        self._min_stage = min_stage
        self._max_stage = max_stage
        # 则阶数self.num_stages为5阶
        self.num_stages = max_stage - min_stage + 1
        if self.num_stages > 1:
            assert len(scales) == len(strides) == self.num_stages, \
                "The num_stages (%d) must match number of scales (%d) and strides (%d)" \
                % (self.num_stages, len(scales), len(strides))
        # 现在支持的最大batch size为1
        self._max_batch = 1  # currently only support batch size = 1
        self._num_sample = num_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        # 这里的RCNNTargetGenerator,表示的就是得到的RPN的结果后应该如何生成RCNN使用的target
        # (TO_DO)这里在使用到的时候再看
        self._target_generator = {RCNNTargetGenerator(self.num_class)}
        # self._additional_output在Mask RCNN中使用,用于存储中间变量
        self._additional_output = additional_output
        with self.name_scope():
            # 这里生成RPN模块
            self.rpn = RPN(
                channels=rpn_channel, strides=strides, base_size=base_size,
                scales=scales, ratios=ratios, alloc_size=alloc_size,
                clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
                train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms,
                test_post_nms=rpn_test_post_nms, min_size=rpn_min_size,
                multi_level=self.num_stages > 1)
            # (TODO)RCNNTargetSampler----用于采样RCNN训练中的正负样本
            self.sampler = RCNNTargetSampler(
                num_image=self._max_batch, num_proposal=rpn_train_post_nms,
                num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
                pos_ratio=pos_ratio, max_num_gt=max_num_gt)

    @property
    def target_generator(self):
        """Returns stored target generator

        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator

        """
        return list(self._target_generator)[0]

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.
        reuse_weights : dict
            A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
            or a list of [name0, name1,...] if class names don't change.
            This allows the new predictor to reuse the
            previously trained weights specified.

        Example
        -------
        >>> net = gluoncv.model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
        >>> # use direct name to name mapping to reuse weights
        >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        >>> # or use interger mapping, person is the 14th category in VOC
        >>> net.reset_class(classes=['person'], reuse_weights={0:14})
        >>> # you can even mix them
        >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
        >>> # or use a list of string if class name don't change
        >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        super(FasterRCNN, self).reset_class(classes, reuse_weights)
        self._target_generator = {RCNNTargetGenerator(self.num_class)}

    def _pyramid_roi_feats(self, F, features, rpn_rois, roi_size, strides, roi_mode='align',
                           eps=1e-6):
        """Assign rpn_rois to specific FPN layers according to its area
           and then perform `ROIPooling` or `ROIAlign` to generate final
           region proposals aggregated features.
        Parameters
        ----------
        features : list of mx.ndarray or mx.symbol
            Features extracted from FPN base network
        rpn_rois : mx.ndarray or mx.symbol
            (N, 5) with [[batch_index, x1, y1, x2, y2], ...] like
        roi_size : tuple
            The size of each roi with regard to ROI-Wise operation
            each region proposal will be roi_size spatial shape.
        strides : tuple e.g. [4, 8, 16, 32]
            Define the gap that ori image and feature map have
        roi_mode : str, default is align
            ROI pooling mode. Currently support 'pool' and 'align'.
        Returns
        -------
        Pooled roi features aggregated according to its roi_level
        """
        max_stage = self._max_stage
        if self._max_stage > 5:  # do not use p6 for RCNN
            max_stage = self._max_stage - 1
        _, x1, y1, x2, y2 = F.split(rpn_rois, axis=-1, num_outputs=5)
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        roi_level = F.floor(4 + F.log2(F.sqrt(w * h) / 224.0 + eps))
        roi_level = F.squeeze(F.clip(roi_level, self._min_stage, max_stage))
        # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
        # roi_level_sorted_args = F.argsort(roi_level, is_ascend=True)
        # roi_level = F.sort(roi_level, is_ascend=True)
        # rpn_rois = F.take(rpn_rois, roi_level_sorted_args, axis=0)
        pooled_roi_feats = []
        for i, l in enumerate(range(self._min_stage, max_stage + 1)):
            # Pool features with all rois first, and then set invalid pooled features to zero,
            # at last ele-wise add together to aggregate all features.
            if roi_mode == 'pool':
                pooled_feature = F.ROIPooling(features[i], rpn_rois, roi_size, 1. / strides[i])
            elif roi_mode == 'align':
                pooled_feature = F.contrib.ROIAlign(features[i], rpn_rois, roi_size,
                                                    1. / strides[i],
                                                    sample_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(roi_mode))
            pooled_feature = F.where(roi_level == l, pooled_feature, F.zeros_like(pooled_feature))
            pooled_roi_feats.append(pooled_feature)
        # Ele-wise add to aggregate all pooled features
        pooled_roi_feats = F.ElementWiseSum(*pooled_roi_feats)
        # Sort all pooled features by asceding order
        # [2,2,..,3,3,...,4,4,...,5,5,...]
        # pooled_roi_feats = F.take(pooled_roi_feats, roi_level_sorted_args)
        # pooled roi feats (B*N, C, 7, 7), N = N2 + N3 + N4 + N5 = num_roi, C=256 in ori paper
        return pooled_roi_feats

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Faster-RCNN network.

        The behavior during training and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """

        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        feat = self.features(x)
        if not isinstance(feat, (list, tuple)):
            feat = [feat]

        # RPN proposals
        if autograd.is_training():
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = \
                self.rpn(F.zeros_like(x), *feat)
            rpn_box, samples, matches = self.sampler(rpn_box, rpn_score, gt_box)
        else:
            _, rpn_box = self.rpn(F.zeros_like(x), *feat)

        # create batchid for roi
        num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms
        with autograd.pause():
            # roi_batchid = F.arange(0, self._max_batch, repeat=num_roi)
            roi_batchid = F.arange(0, self._max_batch)
            roi_batchid = F.repeat(roi_batchid, num_roi)
            # remove batch dim because ROIPooling require 2d input
            rpn_roi = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)
            rpn_roi = F.stop_gradient(rpn_roi)

        if self.num_stages > 1:
            # using FPN
            pooled_feat = self._pyramid_roi_feats(F, feat, rpn_roi, self._roi_size,
                                                  self._strides, roi_mode=self._roi_mode)
        else:
            # ROI features
            if self._roi_mode == 'pool':
                pooled_feat = F.ROIPooling(feat[0], rpn_roi, self._roi_size, 1. / self._strides)
            elif self._roi_mode == 'align':
                pooled_feat = F.contrib.ROIAlign(feat[0], rpn_roi, self._roi_size,
                                                 1. / self._strides, sample_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        # RCNN prediction
        if self.top_features is not None:
            top_feat = self.top_features(pooled_feat)
        else:
            top_feat = pooled_feat
        if self.box_features is None:
            box_feat = F.contrib.AdaptiveAvgPooling2D(top_feat, output_size=1)
        else:
            box_feat = self.box_features(top_feat)
        cls_pred = self.class_predictor(box_feat)
        box_pred = self.box_predictor(box_feat)
        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_class + 1))
        # box_pred (B * N, C * 4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((self._max_batch, num_roi, self.num_class, 4))

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            if self._additional_output:
                return (cls_pred, box_pred, rpn_box, samples, matches,
                        raw_rpn_score, raw_rpn_box, anchors, top_feat)
            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors)

        # cls_ids (B, N, C), scores (B, N, C)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
        cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        # box_pred (B, N, C, 4) -> (B, C, N, 4)
        box_pred = box_pred.transpose((0, 2, 1, 3))

        # rpn_boxes (B, N, 4) -> B * (1, N, 4)
        rpn_boxes = _split(rpn_box, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
        # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
        cls_ids = _split(cls_ids, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
        scores = _split(scores, axis=0, num_outputs=self._max_batch, squeeze_axis=True)
        # box_preds (B, C, N, 4) -> B * (C, N, 4)
        box_preds = _split(box_pred, axis=0, num_outputs=self._max_batch, squeeze_axis=True)

        # per batch predict, nms, each class has topk outputs
        results = []
        for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
            # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
            bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
            # res (C, N, 6)
            res = F.concat(*[cls_id, score, bbox], dim=-1)
            if self.force_nms:
                # res (1, C*N, 6), to allow cross-catogory suppression
                res = res.reshape((1, -1, 0))
            # res (C, self.nms_topk, 6)
            res = F.contrib.box_nms(
                res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
                id_index=0, score_index=1, coord_start=2, force_suppress=self.force_nms)
            # res (C * self.nms_topk, 6)
            res = res.reshape((-3, 0))
            results.append(res)

        # result B * (C * topk, 6) -> (B, C * topk, 6)
        result = F.stack(*results, axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        if self._additional_output:
            return ids, scores, bboxes, feat
        return ids, scores, bboxes


def get_faster_rcnn(name, dataset, pretrained=False, ctx=mx.cpu(),
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return faster rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = FasterRCNN(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('faster_rcnn', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    else:
        for v in net.collect_params().values():
            try:
                v.reset_ctx(ctx)
            except ValueError:
                pass
    return net


def faster_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.14,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)


def faster_rcnn_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    # resnet50作为backbone,这里是gluoncv更新过后的resnet50
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    # 组成FPN的特征提取层,这里只输出了[P2, P3, P4, P5]
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_fpn_bn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                         **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_bn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                                norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs, **kwargs)
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=True, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    # 1 Conv 1 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    box_features.add(nn.Conv2D(256, 3, padding=1),
                     SyncBatchNorm(**gluon_norm_kwargs),
                     nn.Activation('relu'),
                     nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    train_patterns = '(?!.*moving)'  # excluding symbol bn moving mean and var
    return get_faster_rcnn(
        name='fpn_bn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=(640, 800), max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_resnet50_v1b_custom(classes, transfer=None, pretrained_base=True,
                                    pretrained=False, **kwargs):
    r"""Faster RCNN model with resnet50_v1b base network on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        from ..resnetv1b import resnet50_v1b
        base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                    use_global_stats=True, **kwargs)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv',
                                   '.*layers(2|3|4)_conv'])
        return get_faster_rcnn(
            name='resnet50_v1b', dataset='custom', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
            rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
            num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=300,
            **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('faster_rcnn_resnet50_v1b_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def faster_rcnn_resnet101_v1d_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet101_v1d_voc(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet101_v1d
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet101_v1d
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.14,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_fpn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet101_v1d
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    # base_network为resnet101_v1d
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    # 这里填入FPN的模块
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet101_v1d', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_resnet101_v1d_custom(classes, transfer=None, pretrained_base=True,
                                     pretrained=False, **kwargs):
    r"""Faster RCNN model with resnet101_v1d base network on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        from ..resnetv1b import resnet101_v1d
        base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                     use_global_stats=True, **kwargs)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv',
                                   '.*layers(2|3|4)_conv'])
        return get_faster_rcnn(
            name='resnet101_v1d', dataset='custom', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            short=600, max_size=1000, train_patterns=train_patterns,
            nms_thresh=0.3, nms_topk=400, post_nms=100,
            roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
            rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
            rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
            rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
            num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=300,
            **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('faster_rcnn_resnet101_v1d_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net
