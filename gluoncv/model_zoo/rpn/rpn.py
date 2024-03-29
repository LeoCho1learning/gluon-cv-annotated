"""Region Proposal Networks Definition."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn

from .anchor import RPNAnchorGenerator
from .proposal import RPNProposal


class RPN(gluon.HybridBlock):
    r"""Region Proposal Network.

    Parameters
    ----------
    channels : int
        Channel number used in convolutional layers.
    strides : int or tuple of ints
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    clip : float
        Clip bounding box target to this value.
    nms_thresh : float
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int
        Filter top proposals before NMS in training.
    train_post_nms : int
        Return top proposal results after NMS in training.
    test_pre_nms : int
        Filter top proposals before NMS in testing.
    test_post_nms : int
        Return top proposal results after NMS in testing.
    min_size : int
        Proposals whose size is smaller than ``min_size`` will be discarded.
        multi_level : boolean
        Whether to extract feature from multiple level. This is used in FPN.

    """
    '''
    channels----RPN中使用的卷积核数
    strides----(4, 8, 16, 32, 64), 这里其实是输出了4个stage,64这一层是怎么来的(TO_DO)
    base_size, scales, ratios, alloc_size----设置anchor时会使用得到的参数
    clip----Clip bounding box target to this value, 4.14(TO_DO)没有明白这里为什么要设置成4.14
    nms_thresh----rpn中nms使用的iou阈值
    train_pre_nms, train_post_nms, test_pre_nms,test_post_nms----nms过程中的相关参数
    min_size----rpn最后输出框的最小尺寸,当nms之后需要输出的框的尺寸小于这个尺寸时,
    直接取消掉, 1
    multi_level----指示是否是多层输出,示例中因为是FPN,所以为true
    '''
    def __init__(self, channels, strides, base_size, scales, ratios, alloc_size,
                 clip, nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, min_size, multi_level=False, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self._nms_thresh = nms_thresh
        self._multi_level = multi_level
        self._train_pre_nms = max(1, train_pre_nms)
        self._train_post_nms = max(1, train_post_nms)
        self._test_pre_nms = max(1, test_pre_nms)
        self._test_post_nms = max(1, test_post_nms)
        num_stages = len(scales)
        with self.name_scope():
            if self._multi_level:
                asz = alloc_size
                self.anchor_generator = nn.HybridSequential()
                # scale在示例中为(2, 4, 8, 16, 32),这里是RPN生成anchor所必须用到的参数
                # 这里的循环相当于是每一阶的特征图的每一个像素都有三个anchor,不同阶的scale大小不一样
                for _, st, s in zip(range(num_stages), strides, scales):
                    stage_anchor_generator = RPNAnchorGenerator(st, base_size, ratios, s, asz)
                    self.anchor_generator.add(stage_anchor_generator)
                    asz = max(asz[0] // 2, 16)
                    asz = (asz, asz)  # For FPN, We use large anchor presets
                # 这里的anchor_depth就相当于是每个像素生成的anchor的数量
                anchor_depth = self.anchor_generator[0].num_depth
                # 生成RPN的预测head
                # 这里的RPN_head的channels在示例中是1024,(TO_DO)为什么是1024,这里没有
                # 将RPN_head放入到循环中,应该是进行了参数共享
                self.rpn_head = RPNHead(channels, anchor_depth)
            else:
                self.anchor_generator = RPNAnchorGenerator(
                    strides, base_size, ratios, scales, alloc_size)
                anchor_depth = self.anchor_generator.num_depth
                # not using RPNHead to keep backward compatibility with old models
                weight_initializer = mx.init.Normal(0.01)
                self.conv1 = nn.HybridSequential()
                self.conv1.add(nn.C onv2D(channels, 3, 1, 1, weight_initializer=weight_initializer),
                               nn.Activation('relu'))
                self.score = nn.Conv2D(anchor_depth, 1, 1, 0, weight_initializer=weight_initializer)
                self.loc = nn.Conv2D(anchor_depth * 4, 1, 1, 0,
                                     weight_initializer=weight_initializer)
            # 生成proposal模块,用于将RPN的预测结果转换为前景与背景
            self.region_proposer = RPNProposal(clip, min_size, stds=(1., 1., 1., 1.))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, img, *x):
        """Forward RPN.

        The behavior during training and inference is different.

        Parameters
        ----------
        img : mxnet.nd.NDArray or mxnet.symbol
            The original input image.
        x : mxnet.nd.NDArray or mxnet.symbol(s)
            Feature tensor(s).

        Returns
        -------
        (rpn_score, rpn_box)
            Returns predicted scores and regions which are candidates of objects.

        """
        if autograd.is_training():
            pre_nms = self._train_pre_nms
            post_nms = self._train_post_nms
        else:
            pre_nms = self._test_pre_nms
            post_nms = self._test_post_nms
        anchors = []
        rpn_pre_nms_proposals = []
        raw_rpn_scores = []
        raw_rpn_boxes = []
        if self._multi_level:
            # Generate anchors in [P2, P3, P4, P5, P6] order
            for i, feat in enumerate(x):
                ag = self.anchor_generator[i]
                anchor = ag(feat)
                rpn_score, rpn_box, raw_rpn_score, raw_rpn_box = \
                    self.rpn_head(feat)
                rpn_pre = self.region_proposer(anchor, rpn_score,
                                               rpn_box, img)
                anchors.append(anchor)
                rpn_pre_nms_proposals.append(rpn_pre)
                raw_rpn_scores.append(raw_rpn_score)
                raw_rpn_boxes.append(raw_rpn_box)
            rpn_pre_nms_proposals = F.concat(*rpn_pre_nms_proposals, dim=1)
            raw_rpn_scores = F.concat(*raw_rpn_scores, dim=1)
            raw_rpn_boxes = F.concat(*raw_rpn_boxes, dim=1)
        else:
            x = x[0]
            anchors = self.anchor_generator(x)
            x = self.conv1(x)
            raw_rpn_scores = self.score(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
            rpn_scores = F.sigmoid(F.stop_gradient(raw_rpn_scores))
            raw_rpn_boxes = self.loc(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 4))
            rpn_boxes = F.stop_gradient(raw_rpn_boxes)
            rpn_pre_nms_proposals = self.region_proposer(
                anchors, rpn_scores, rpn_boxes, img)

        # Non-maximum suppression
        with autograd.pause():
            tmp = F.contrib.box_nms(rpn_pre_nms_proposals, overlap_thresh=self._nms_thresh,
                                    topk=pre_nms, coord_start=1, score_index=0, id_index=-1,
                                    force_suppress=True)

            # slice post_nms number of boxes
            result = F.slice_axis(tmp, axis=1, begin=0, end=post_nms)
            rpn_scores = F.slice_axis(result, axis=-1, begin=0, end=1)
            rpn_boxes = F.slice_axis(result, axis=-1, begin=1, end=None)

        if autograd.is_training():
            # return raw predictions as well in training for bp
            return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes, anchors
        return rpn_scores, rpn_boxes

    def get_test_post_nms(self):
        return self._test_post_nms


class RPNHead(gluon.HybridBlock):
    r"""Region Proposal Network Head.

    Parameters
    ----------
    channels : int
        Channel number used in convolutional layers.
    anchor_depth : int
        Each FPN stage have one scale and three ratios,
        we can compute anchor_depth = len(scale) \times len(ratios)
    """

    def __init__(self, channels, anchor_depth, **kwargs):
        super(RPNHead, self).__init__(**kwargs)
        weight_initializer = mx.init.Normal(0.01)
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            self.conv1.add(nn.Conv2D(channels, 3, 1, 1, weight_initializer=weight_initializer),
                           nn.Activation('relu'))
            # use sigmoid instead of softmax, reduce channel numbers
            # Note : that is to say, if use softmax here,
            # then the self.score will anchor_depth*2 output channel
            # 这里选择使用sigmoid作为RPN的分类head的激活函数,RPN只需要负责二分类,因此这里
            # 的通道数等于anchor的数量即可
            self.score = nn.Conv2D(anchor_depth, 1, 1, 0, weight_initializer=weight_initializer)
            self.loc = nn.Conv2D(anchor_depth * 4, 1, 1, 0, weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Forward RPN Head.

        This HybridBlock will generate predicted values for cls and box.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            Feature tensor. With (1, C, H, W) shape

        Returns
        -------
        (rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes)
            Returns predicted scores and regions.

        """
        # 3x3 conv with relu activation
        x = self.conv1(x)
        # (1, C, H, W)->(1, 9, H, W)->(1, H, W, 9)->(1, H*W*9, 1)
        raw_rpn_scores = self.score(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
        # (1, H*W*9, 1)
        rpn_scores = F.sigmoid(F.stop_gradient(raw_rpn_scores))
        # (1, C, H, W)->(1, 36, H, W)->(1, H, W, 36)->(1, H*W*9, 4)
        raw_rpn_boxes = self.loc(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 4))
        # (1, H*W*9, 1)
        rpn_boxes = F.stop_gradient(raw_rpn_boxes)
        # return raw predictions as well in training for bp
        return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes
