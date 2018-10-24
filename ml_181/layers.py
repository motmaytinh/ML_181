import numpy as np

def conv_naive_forward(x, w, b, conv_param):

    pad = conv_param["pad"]
    stride = conv_param["stride"]

    x_padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = (W - WW + 2 * pad) // stride + 1
    Wout = (H - HH + 2 * pad) // stride + 1
    out = np.zeros((N, F, Hout, Wout))
    for idx_image, each_image in enumerate(x_padded):
        for i_H in range(Hout):
            for i_W in range(Wout):
                im_patch = each_image[:, i_H * stride:i_H * stride + HH,
                                      i_W * stride:i_W * stride + WW]
                scores = (w * im_patch).sum(axis=(1, 2, 3)) + b

                out[idx_image, :, i_H, i_W] = scores

    cache = (x, w, b, conv_param)
    return out, cache


def max_pool_naive_forward(x, pool_param):
    (N, C, H, W) = x.shape

    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, Hout, Wout))

    for idx_image, each_image in enumerate(x):
        for i_H in range(Hout):
            for i_W in range(Wout):

                each_window_channels = each_image[:, i_H*stride: i_H*stride + pool_height,
                                                i_W*stride: i_W*stride + pool_width ]

                out[idx_image, :, i_H, i_W] = each_window_channels.max(axis = (1,2)) # maxpooling


    cache = (x, pool_param)

    return out, cache
