import pdb
import cv2
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
from functools import partial
import matlab.engine

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))

def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)

def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i+window_size, j:j+window_size, c]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr**2)
    # assert np.isnan(ssq/total)
    return ssq / total

def compare_ncc(x, y):
    return np.mean((x-np.mean(x)) * (y-np.mean(y))) / (np.std(x) * np.std(y))


def PCQI(img1,img2, eng):
    # img1 = tensor2im(img1)

    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    # img2 = tensor2im(img2)
    img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    # import ipdb;ipdb.set_trace()

    # eng = matlab.engine.start_matlab()
    res = eng.PCQI(matlab.double(img1.tolist()),matlab.double(img2.tolist()))
    return res


def quality_assess(X, Y ,Z):
    # Y: correct; X: estimate
    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))
    lmse = local_error(Y, X, 20, 10)
    ncc = compare_ncc(Y, X)
    eng = matlab.engine.start_matlab()
    pcqi = PCQI(Z,X,eng)
    return {'PSNR':psnr, 'SSIM': ssim, 'LMSE': lmse, 'NCC': ncc ,'PCQI':pcqi}

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy.astype(np.uint8)




if __name__=="__main__":
    import torch
    a = torch.randn([1,3,256,256])
    b = torch.randn([1,3,256,256])
    print(pcqi(a,b))
