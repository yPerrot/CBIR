# -*- coding: utf-8 -*-

from __future__ import print_function

from DB import Database
from evaluate import distance, evaluate_class, infer, KNN, get_cls

from six.moves import cPickle
import numpy as np
import scipy.misc
import itertools
import os

import imageio

# configs for histogram
n_bin   = 12        # histogram bins
n_slice = 3         # slice image
h_type  = 'region'  # global or region
d_type  = 'd1'      # distance type

depth   = 3         # retrieved depth, set to None will count the ap for whole database

''' MMAP
     depth
      depthNone, region,bin12,slice3, distance=d1, MMAP 0.273745840034
      depth100,  region,bin12,slice3, distance=d1, MMAP 0.406007856783
      depth30,   region,bin12,slice3, distance=d1, MMAP 0.516738512679
      depth10,   region,bin12,slice3, distance=d1, MMAP 0.614047666604
      depth5,    region,bin12,slice3, distance=d1, MMAP 0.650125
      depth3,    region,bin12,slice3, distance=d1, MMAP 0.657166666667
      depth1,    region,bin12,slice3, distance=d1, MMAP 0.62

     (exps below use depth=None)
     
     d_type
      global,bin6,d1,MMAP 0.242345913685
      global,bin6,cosine,MMAP 0.184176505586

     n_bin
      region,bin10,slice4,d1,MMAP 0.269872790396
      region,bin12,slice4,d1,MMAP 0.271520862017

      region,bin6,slcie3,d1,MMAP 0.262819311357
      region,bin12,slice3,d1,MMAP 0.273745840034

     n_slice
      region,bin12,slice2,d1,MMAP 0.266076627332
      region,bin12,slice3,d1,MMAP 0.273745840034
      region,bin12,slice4,d1,MMAP 0.271520862017
      region,bin14,slice3,d1,MMAP 0.272386552594
      region,bin14,slice5,d1,MMAP 0.266877181379
      region,bin16,slice3,d1,MMAP 0.273716788003
      region,bin16,slice4,d1,MMAP 0.272221031804
      region,bin16,slice8,d1,MMAP 0.253823360098

     h_type
      region,bin4,slice2,d1,MMAP 0.23358615622
      global,bin4,d1,MMAP 0.229125435746
'''

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Color(object):

  def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img color histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        n_bin    : number of bins for each channel
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size n_bin ** channel
        type == 'region'
          a numpy array with size n_slice * n_slice * (n_bin ** channel)
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input)
    height, width, channel = img.shape
    bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel
  
    if type == 'global':
      hist = self._count_hist(img, n_bin, bins, channel)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin ** channel))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _count_hist(self, input, n_bin, bins, channel):
    img = input.copy()
    bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
    hist = np.zeros(n_bin ** channel)
  
    # cluster every pixels
    for idx in range(len(bins)-1):
      img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
    # add pixels into bins
    height, width, _ = img.shape
    for h in range(height):
      for w in range(width):
        b_idx = bins_idx[tuple(img[h,w])]
        hist[b_idx] += 1
  
    return hist
  
  
  def make_samples(self, db, sample_name="", verbose=True):
    if h_type == 'global':
      sample_cache = "histogram_cache-{}-n_bin{}-name{}".format(h_type, n_bin, sample_name)
    elif h_type == 'region':
      sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}-name{}".format(h_type, n_bin, n_slice, sample_name)
    
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb"))
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb"))
    return samples


if __name__ == "__main__":
  color = Color()

  # Create my samples
  db = Database("database\\train")
  print("Train databse created.")
  samples = color.make_samples(db, sample_name="train")
  print("Train samples created.")

  test = Database("database\dev")
  print("Test databse created.")
  sample_test = color.make_samples(test, sample_name="dev")
  print("Test samples created.")

  # Find class for each image of my test DB and verify the result
  nb_good_classification = 0
  for img_test in sample_test:
    _, resultes = infer(img_test, samples)
    real_cls = KNN(resultes, db.get_class())

    nb_good_classification += get_cls(img_test['cls']) == get_cls(real_cls)

  print("\n{}/{}".format(nb_good_classification, len(sample_test)))