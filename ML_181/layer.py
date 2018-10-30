def neural_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None
    
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db