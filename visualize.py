from scipy import misc
import torch
import numpy as np
from skimage import io

epoch = 248
seq_len = 8
true_path = f'results/visualizations/trues_{8}.npy'
pred_path = f'results/visualizations/preds_{epoch}.npy'
save_path = f'results/visualizations/{epoch}.png'

trues = np.load(true_path)
preds = np.load(pred_path)
print(np.sum(np.abs(trues - preds)) / trues.shape[1]*trues.shape[0])
row_size = trues.shape[1]
vis_array = np.zeros((trues.shape[0]*2*256, seq_len*128, 3), dtype=np.uint8)
for idx, (true, pred) in enumerate(zip(trues, preds)):
    true = np.transpose(true, axes=[0, 2, 3, 1])
    pred = np.transpose(pred, axes=[0, 2, 3, 1])
    true = np.maximum(true, 0.0)
    pred = np.maximum(pred, 0.0)
    true = np.minimum(true, 1.0)
    pred = np.minimum(pred, 1.0)
    pred = pred * 255
    true = true * 255
    vis_array[idx*2*256:(idx*2+1)*256] = np.hstack((t for t in true)).astype(np.uint8)
    vis_array[(idx*2+1)*256:(idx*2+2)*256] = np.hstack((p for p in pred)).astype(np.uint8)

image = io.imsave(save_path, vis_array)

