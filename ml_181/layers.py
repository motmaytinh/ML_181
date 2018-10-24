import numpy as np

def conv_naive_forward(x, w, b, conv_param):

    pad = conv_param["pad"]
    stride = conv_param["stride"]

    x_padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = int((W - WW + 2 * pad) / stride + 1)
    Wout = int((H - HH + 2 * pad) / stride + 1)
    out = np.zeros((N, F, Hout, Wout))
    for i in range(N):
        image = x_padded[i, :, :, :]
        for j in range(Hout):
            for k in range(Wout):
                im_patch = image[:, j * stride:j *
                                 stride + HH, k * stride:k * stride + WW]
                scores = (w * im_patch).sum(axis=(1, 2, 3)) + b
                out[i, :, j, k] = scores

    cache = (x, w, b, conv_param)
    return out, cache
