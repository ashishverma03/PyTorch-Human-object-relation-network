"""Transforms for Human-object Relation Network."""
from __future__ import absolute_import
# import mxnet as mx
# from mxnet.gluon import data as gdata
import torchvision.transforms as gdata
import random
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
# from .. import bbox as tbbox
# from .. import image as timage


def _get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def resize(bbox, in_size, out_size):
    """Resize bouding boxes according to image resize operation.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    return bbox


def flip(bbox, size, flip_x=False, flip_y=False):
    """Flip bounding boxes according to image flipping directions.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2: (width, height).
    flip_x : bool
        Whether flip horizontally.
    flip_y : type
        Whether flip vertically.

    Returns
    -------
    numpy.ndarray
        Flipped bounding boxes with original shape.
    """
    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))
    width, height = size
    bbox = bbox.copy()
    if flip_y:
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    if flip_x:
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax
    return bbox

def random_flip(src, px=0, py=0, copy=False):
    """Randomly flip image along horizontal and vertical with probabilities.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image with HWC format.
    px : float
        Horizontal flip probability [0, 1].
    py : float
        Vertical flip probability [0, 1].
    copy : bool
        If `True`, return a copy of input

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (flip_x, flip_y), records of whether flips are applied.

    """
    # src = torch.from_numpy(src)
    # print("type of src: ",type(src))
    flip_y = np.random.choice([False, True], p=[1-py, py])
    flip_x = np.random.choice([False, True], p=[1-px, px])
    if flip_y:
        src = torch.flip(torch.from_numpy(src), [0])
    if flip_x:
        src = torch.flip(torch.from_numpy(src), [1])
    if copy:
        src = src.copy()
    return src, (flip_x, flip_y)


# # May be removed...see it later
# def imresize(src, w, h, interp=1):
#     """Resize image with OpenCV.

#     This is a duplicate of mxnet.image.imresize for name space consistancy.

#     Parameters
#     ----------
#     src : mxnet.nd.NDArray
#         source image
#     w : int, required
#         Width of resized image.
#     h : int, required
#         Height of resized image.
#     interp : int, optional, default='1'
#         Interpolation method (default=cv2.INTER_LINEAR).

#     out : NDArray, optional
#         The output NDArray to hold the result.

#     Returns
#     -------
#     out : NDArray or list of NDArrays
#         The output of this function.

#     Examples
#     --------
#     >>> import mxnet as mx
#     >>> from gluoncv import data as gdata
#     >>> img = mx.random.uniform(0, 255, (300, 300, 3)).astype('uint8')
#     >>> print(img.shape)
#     (300, 300, 3)
#     >>> img = gdata.transforms.image.imresize(img, 200, 200)
#     >>> print(img.shape)
#     (200, 200, 3)
#     """
#     oh, ow, _ = src.shape
#     interpolation_list = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
#     # return cv2.resize(src, w, h, interpolation=interpolation_list[_get_interp_method(interp, (oh, ow, h, w))])
#     return cv2.resize(src, w, h, interpolation=cv2.INTER_AREA)


def resize_short_within(src, short, max_size, mult_base=1, interp=2):
    """Resizes shorter edge to size but make sure it's capped at maximum size.
    Note: `resize_short_within` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `resize_short_within` to work.
    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly. Also this function will ensure
    the new image will not exceed ``max_size`` even at the longer side.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    short : int
        Resize shorter side to ``short``.
    max_size : int
        Make sure the longer side of new image is smaller than ``max_size``.
    mult_base : int, default is 1
        Width and height are rounded to multiples of `mult_base`.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.
    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1000)
    >>> new_image
    <NDArray 667x1000x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1200)
    >>> new_image
    <NDArray 800x1200x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1200, mult_base=32)
    >>> new_image
    <NDArray 800x1184x3 @cpu(0)>
    """
    h, w, _ = src.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))

    interpolation_list = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
    # return imresize(src, new_w, new_h, interp=_get_interp_method(interp, (h, w, new_h, new_w)))
    return cv2.resize(src, (new_w, new_h), interpolation=interpolation_list[_get_interp_method(interp, (h, w, new_h, new_w))])


class HORelationDefaultTrainTransform(object):
    """Default RODet train transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """


    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size
        self._color_jitter = gdata.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, src, label, box):
        """Apply transform to validation image/label."""
        img = src
        bbox = label
        objbox = box

        # # random crop
        # h, w, _ = img.shape
        # bbox, crop = tbbox.random_crop_with_constraints(bbox, (w, h))
        # x0, y0, w, h = crop
        # img = mx.image.fixed_crop(src, x0, y0, w, h)

        # resize shorter side but keep in max_size
        h, w, _ = img.shape
        # print(img.shape)
        # cv2.imshow("before",img)
        # print("before:",img.shape)
        img = resize_short_within(img, self._short, self._max_size, interp=1)
        # cv2.imwrite('1.png',img)
        # cv2.imshow("after",img)
        # print("after:",img.shape)
        # cv2.waitKey()
        
        bbox = resize(bbox, (w, h), (img.shape[1], img.shape[0]))
        objbox = resize(objbox, (w, h), (img.shape[1], img.shape[0]))

        # color jitter
        # img = self._color_jitter(img)

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = random_flip(img, px=0.5)
        bbox = flip(bbox, (w, h), flip_x=flips[0])

        # img = mx.nd.image.to_tensor(img)
        img = np.array(img)
        mytransform = gdata.Compose([gdata.ToTensor(),gdata.Normalize(mean=self._mean,std=self._std)])
        # mytransform = gdata.Compose([gdata.Normalize(mean=self._mean,std=self._std)])
        img = mytransform(img)
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        # img = gdata.Normalize()
        return img, bbox.astype('float32'), objbox.astype('float32')


class HORelationDefaultValTransform(object):
    """Default RODet validation transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, label, box):
        """Apply transform to validation image/label."""

        objbox = box

        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        bbox = resize(label, (w, h), (img.shape[1], img.shape[0]))
        objbox = resize(objbox, (w, h), (img.shape[1], img.shape[0]))

        # img = mx.nd.image.to_tensor(img)
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        transform = gdata.Compose([gdata.ToTensor(),gdata.Normalize(mean=self._mean,std=self._std)])
        img = transform(img)
        return img, bbox.astype('float32'), objbox.astype('float32')


class HORelationDefaultVisTransform(object):
    """Default RODet Visualization transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, label, box):
        """Apply transform to validation image/label."""
        objbox = box
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = resize_short_within(src, self._short, self._max_size, interp=1)
        ori_img = img.astype('uint8')
        # no scaling ground-truth, return image scaling ratio instead
        bbox = resize(label, (w, h), (img.shape[1], img.shape[0]))
        objbox = resize(objbox, (w, h), (img.shape[1], img.shape[0]))

        # img = mx.nd.image.to_tensor(img)
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        transform = gdata.Compose([gdata.ToTensor(),gdata.Normalize(mean=self._mean,std=self._std)])
        img = transform(img)
        return img, bbox.astype('float32'), objbox.astype('float32'), ori_img
