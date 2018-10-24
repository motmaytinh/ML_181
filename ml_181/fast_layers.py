import numpy as np

def conv_fast_forward(x, w, b, conv_param):

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad']

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    Hout = (H + 2 * pad - HH) // stride + 1
    Wout = (W + 2 * pad - WW) // stride + 1

    H += 2*pad
    W += 2*pad

    shape = (C, HH, WW, N, Hout, Wout)

    #im2col flatten (34x34, 34, 1, 3x34x34, 2x32, 2)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)

    #length in bytes
    strides = x.itemsize * np.array(strides)

    #DOC slide 9: creates a view into the array given the exact strides and shape.
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                  shape=shape, strides=strides)

    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * Hout * Wout)

    # Now all our convolutions are a big matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    res.shape = (F, N, Hout, Wout)

    # transpose to N, F, Hout, Wout
    out = res.transpose(1, 0, 2, 3)

    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_forward_im2col(x, w, b, conv_param):

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad']

    Hout = (H + 2 * pad - filter_height) // stride + 1
    Wout = (W + 2 * pad - filter_width) // stride + 1

    out = np.zeros((N, num_filters, Hout, Wout))
