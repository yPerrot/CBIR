# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import evaluate_class, infer, KNN, get_cls
from DB import Database

from color import Color
from daisy import Daisy

import numpy as np
import itertools
import os


d_type   = 'd1'
depth    = 30

feat_pools = ['color', 'daisy', 'edge', 'gabor', 'hog', 'vgg', 'res']

class FeatureFusion(object):

  def __init__(self, features):
    assert len(features) > 1, "need to fuse more than one feature!"
    self.features = features

  def make_samples(self, db, sample_name="", verbose=False):
    if verbose:
      print("Use features {}".format(" & ".join(self.features)))

    feats = []
    for f_class in self.features:
      feats.append(self._get_feat(db, f_class, sample_name=sample_name))
    samples = self._concat_feat(db, feats)
    return samples

  def _get_feat(self, db, f_class, sample_name=""):
    if f_class == 'color':
      f_c = Color()
    elif f_class == 'daisy':
      f_c = Daisy()
    return f_c.make_samples(db, verbose=False, sample_name=sample_name)

  def _concat_feat(self, db, feats):
    samples = feats[0]
    delete_idx = []
    for idx in range(len(samples)):
      for feat in feats[1:]:
        feat = self._to_dict(feat)
        key = samples[idx]['img']
        if key not in feat:
          delete_idx.append(idx)
          continue
        assert feat[key]['cls'] == samples[idx]['cls']
        samples[idx]['hist'] = np.append(samples[idx]['hist'], feat[key]['hist'])
    for d_idx in sorted(set(delete_idx), reverse=True):
      del samples[d_idx]
    if delete_idx != []:
      print("Ignore %d samples" % len(set(delete_idx)))

    return samples

  def _to_dict(self, feat):
    ret = {}
    for f in feat:
      ret[f['img']] = {
        'cls': f['cls'],
        'hist': f['hist']
      }
    return ret

if __name__ == "__main__":

  fusion = FeatureFusion(features=['color', 'daisy'])

  # Create my samples
  db = Database("database\\train")
  print("Train databse created.")
  samples = fusion.make_samples(db, sample_name="train")
  print("Train samples created.")

  test = Database("database\dev")
  print("Test databse created.")
  sample_test = fusion.make_samples(test, sample_name="dev")
  print("Test samples created.")

  # Find class for each image of my test DB and verify the result
  nb_good_classification = 0
  for img_test in sample_test:
    _, resultes = infer(img_test, samples)
    real_cls = KNN(resultes, db.get_class())

    nb_good_classification += get_cls(img_test['cls']) == get_cls(real_cls)

  print("\n{}/{}".format(nb_good_classification, len(sample_test)))