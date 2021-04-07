import numpy as np
from skimage.transform import resize as imresize
import scipy
import matplotlib.pyplot as plt
import pickle

from scores import learning_info
from matplotlib.image import imread

def monitor(x):
    if x<=10 or(x<50 and x%10==0) or x%50==0:
        print(x)

def prep_x_resize_vgg(images, s):
    N = len(images)
    X_train = np.zeros((N,3,s,s),dtype="float32")
    for i in range(N):
        monitor(i)
        im = images[i]
        im = imresize(im,(s,s))
        im = prep_image_vgg(im)
        X_train[i,:,:,:] = im
    return X_train

def prep_image_vgg(im):
    im = np.float32(im)

    im *= 255.0
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def prep_Y_for_net(y):
    n = len(y)
    arr = np.zeros((n,1))
    for i in range(len(y)):
        arr[i,0] = y[i]
    return arr
        
def load_demo_data(f_pvoc):
    images = []
    keys = []
    urls = []
    mus = []

    keys = list(learning_info.keys())

    for i in keys[:400]:
        print(i)
        images.append(imread(f_pvoc + learning_info[i]['url']))
        keys.append(i)
        urls.append(learning_info[i]['url'])
        mus.append(learning_info[i]['mu'])

    return (images, mus, keys, urls)

def remove_small_images(images,mus,keys):
    cleaned_images = []
    cleaned_mus = []
    cleaned_keys = []

    for i,im in enumerate(images):
        if im.shape[0]<200:
            print(i)
        else:
            cleaned_images.append(im)
            cleaned_mus.append(mus[i])
            cleaned_keys.append(keys[i])
    assert(len(cleaned_images) == len(cleaned_mus))
    assert(len(cleaned_images) == len(cleaned_keys) == len(cleaned_mus))

    return cleaned_images, cleaned_mus, cleaned_keys


def visual_correlation(M,X,Y):
    Y_pred = M.predict(X)
    fig,ax = plt.subplots()
    ax.scatter(Y, Y_pred)
    ax.set_xlim((0,50))
    ax.set_ylim((0,50))
    ax.set_aspect('equal')
    ax.set_xlabel('Observer rating')
    ax.set_ylabel('Model rating')
    Y_f = Y.flatten()
    Y_pred_f = Y_pred.flatten()
    r,p = scipy.stats.pearsonr(Y_f, Y_pred_f)
    r = np.round(r,2)
    print("R =",r)

