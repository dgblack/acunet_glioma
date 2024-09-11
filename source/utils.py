import torch
import numpy as np

def combine_ptdictdata(*dataf):
  datas = [torch.load(f) for f in dataf]
  keys = set(datas[0].keys())
  for d in datas:
    keys.intersection_update(d.keys())

  out = {k: [] for k in keys}
  for k in keys:
    for d in datas:
      out[k].append(d[k])

  for k, v in out.items():
    if torch.is_tensor(v[0]):
      out[k] = torch.concat(v, dim=0)
    if isinstance(v[0], np.ndarray):
      out[k] = np.concatenate(v, axis=0)
  return out

def print_ptdictdata_shape(data):
  for k, v in data.items(): print(k, v.shape)

def mask_ptdictdata_shape(data):
  for k, v in data.items(): print(k, v.shape)