import cv2
import numpy as np
from numpy import linalg as LA
from typing import List
import matplotlib.pyplot as plt

# by Shira Baron -208761452

# Lucas Kanade optical ﬂow ---------------------------------------------------

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5)-> (np.ndarray, np.ndarray):
    # convert to grayscale if it is rgb
    if (not isgray(im1)):
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if (not isgray(im2)):
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # kernels to derivate x and y
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    # derivate and blur
    Ix = cv2.filter2D(im2, -1, kernelx)
    Ix = cv2.GaussianBlur(Ix, (5, 5), 0)
    Iy = cv2.filter2D(im2, -1, kernely)
    Iy = cv2.GaussianBlur(Iy, (5, 5), 0)
    # calculate It
    It = im2 - im1

    w = int(win_size / 2)
    xyList = []
    duvList = []
    for i in range(w, im1.shape[0] - w, step_size):
        for j in range(w, im1.shape[1] - w, step_size):
            ix = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            iy = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            it = It[i - w:i + w + 1, j - w:j + w + 1].flatten()
            a = ix.reshape(ix.shape[0], 1)
            B = it.reshape(it.shape[0], 1) * (-1)
            A = np.full((win_size ** 2, 2), 0)
            A[:, 0] = ix
            A[:, 1] = iy
            # the purpuse is to find vx, vy
            # before we need to check if ATA has self-value
            # 1. At*A
            At = np.transpose(A)
            ATA = At.dot(A)
            # find eginvalue
            eginvalue, v = LA.eig(ATA)
            eginvalue = np.sort(eginvalue)
            lambda1 = eginvalue[-1]
            lambda2 = eginvalue[-2]
            # check if lambdas are correct
            if (lambda2 > 1) and (lambda1 / lambda2) < 100:
                # V = (ATA)*-1*AT*B
                ATAreverse = LA.inv(ATA)
                z=At.dot(B)
                v = ATAreverse.dot(z)
                xyList.append([j, i])
                duvList.append(v)
    xyList = np.asarray(xyList)
    duvList = np.asarray(duvList)
    return (xyList, duvList)

# stackoverflow
def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

# Gaussian Pyramids ----------------------------------------------------

def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    pyrLst = []
    pyrLst.append(img)
    gauss=cv2.getGaussianKernel(5,sigma = (0.3  + 0.8))
    gauss = np.outer(gauss, gauss.transpose())
    for i in range(1, levels):
        Imgt = cv2.filter2D(pyrLst[i-1], -1, gauss)
        Imgt = Imgt[:: 2, :: 2]
        pyrLst.append(Imgt)
    return pyrLst


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if (len(img.shape) == 2):
        w = img.shape[0]
        h = img.shape[1]
        newImage = np.full((2 * w, 2 * h), 0, dtype=img.dtype)
        newImage = newImage.astype(np.float)
        newImage[::2, ::2] = img
    if (len(img.shape) == 3):
        w, h, z = img.shape
        newImage = np.full((2 * w, 2 * h, z), 0, dtype=img.dtype)
        newImage = newImage.astype(np.float)
        newImage[::2, ::2] = img

    gs_k = (gs_k * 4) / gs_k.sum()  # make sure it is 4
    newImage = cv2.filter2D(newImage, -1, gs_k, borderType=cv2.BORDER_DEFAULT)

    return newImage

# laplaceian Pyramids ----------------------------------------------------

def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gaussList = gaussianPyr(img, levels)
    lpList = []
    for i in range(len(gaussList) - 1):
        smaller = gaussList[i + 1]
        exp = gaussExpand(smaller, gaussKernel)
        a = gaussList[i].shape == exp.shape
        if (not a): exp = exp[:-1, :-1]
        newLevel = gaussList[i] - exp
        lpList.append(newLevel)
    lpList.append(gaussList[-1])
    return lpList


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    lpList = lap_pyr[::-1]
    result = lpList[0]
    size = len(lpList)
    for i in range(size - 1):
        exp = gaussExpand(result, gaussKernel)
        a = lpList[i + 1].shape == exp.shape
        if (not a): exp = exp[:-1, :-1]
        result = lpList[i + 1] + exp
    return result

# Blending ----------------------------------------------------

def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):

    NaiveBlend = img_1*mask +img_2*(1-mask)
    # build laplacian pyramids LA and LB from image A and B
    LA = laplaceianReduce(img_1, levels)
    LB= laplaceianReduce(img_2, levels)

    # Build a gaussian pyramid GR from selected region mask
    GR = gaussianPyr(mask,levels)

    LS=[]
    # form a combind pyramid LS from LA and LB using nodes of GR as wights:
    for i in range (levels):
      LS.append(GR[i]*LA[i]+(1-GR[i])*LB[i])

    # collapse the LS pyramid to get the final blended image
    Blended_Image = laplaceianExpand(LS)
    return (NaiveBlend,Blended_Image)
gaussKernel = np.array([[1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445],
                        [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                        [6.49510362, 25.90969361, 41.0435344, 25.90969361, 6.49510362],
                        [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                        [1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445]])
pass

