import sys
sys.path.append("../miqa-seg")
import os
import numpy as np
sys.path.insert(1, os.path.abspath('.'))
import utils
from utils import model_config
import shutil
import matplotlib.pyplot as plt

idx = 0
while True:
    if (os.path.isdir(str(idx))):
        shutil.rmtree(str(idx))
        idx += 1
    else:
        break

N = 5
data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/MIQA"
patients = os.listdir(data_path)
# np.random.shuffle(patients)
# patients = np.array(patients)[np.random.choice(len(patients), size=N, replace=False)]
patients = sorted(patients)[800:800+N]
width = 60
height = width / 2

pat_idx = 0
saved_idx = 0
text = str(N) + '\n'
while True:
    if (saved_idx >= N):
        break

    if (not os.path.isdir(str(saved_idx))):
        os.mkdir(str(saved_idx))
    patient = patients[pat_idx]
    patient_maps, maps_count = utils.load_volume(data_path + "/" + patient)

    available_keys = []
    for key in patient_maps.keys():
        if (len(patient_maps[key]) > 0):
            if (np.sum(patient_maps[key]) > 0):
                if ((key != "CT") and (key != "Background")):
                    available_keys.append(key)
    
    if (len(available_keys) == 0):
        pat_idx += 1
        continue

    selected_key = np.array(available_keys)[np.random.choice(len(available_keys), size=1, replace=False)][0]
    
    CT = patient_maps["CT"]
    segmentation = patient_maps[selected_key]


    first = np.min(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0))
    last = np.max(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0)) + 1
    plot_idx = 0
    for i in range(first, last):
        plt.figure(figsize=(width, height))
        plt.subplot(121)
        plt.imshow(np.fliplr(np.flipud(np.transpose(CT[i, :, :]))), vmin=0, vmax=255, cmap="gray")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.subplot(122)
        plt.imshow(np.fliplr(np.flipud(np.transpose(segmentation[i, :, :]))), vmin=0, vmax=1, cmap="gray")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        plt.savefig(str(saved_idx) + "/cor" + str(plot_idx) + ".png")
        plot_idx += 1
    text += (str(last - first - 1) + ',')

    first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0))
    last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0)) + 1
    plot_idx = 0
    for i in range(first, last):
        plt.figure(figsize=(width, height))
        plt.subplot(121)
        plt.imshow(np.flipud(np.transpose(CT[:, i, :])), vmin=0, vmax=255, cmap="gray")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.subplot(122)
        plt.imshow(np.flipud(np.transpose(segmentation[:, i, :])), vmin=0, vmax=1, cmap="gray")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        plt.savefig(str(saved_idx) + "/sag" + str(plot_idx) + ".png")
        plot_idx += 1
    text += (str(last - first - 1) + ',')

    first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0))
    last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0)) + 1
    plot_idx = 0
    for i in range(first, last):
        plt.figure(figsize=(width, height))
        plt.subplot(121)
        plt.imshow(CT[:, :, i], vmin=0, vmax=255, cmap="gray")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.subplot(122)
        plt.imshow(segmentation[:, :, i], vmin=0, vmax=1, cmap="gray")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        plt.savefig(str(saved_idx) + "/ax" + str(plot_idx) + ".png")
        plot_idx += 1
    text += (str(last - first - 1) + '\n')

    pat_idx += 1
    saved_idx += 1

with open('patientlog.txt', 'w') as f:
    f.write(text)