import sys
sys.path.append("../miqa-seg")
import os
import numpy as np
sys.path.insert(1, os.path.abspath('.'))
from utils import model_config, load_volume
from model import OurNet
import shutil
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.ion()
import tensorflow
from skimage.measure import label 

def getLargestCC(segmentation):
    labels = label(segmentation)
    if (labels.max() == 0):
        return np.zeros(segmentation.shape, np.int32)
    
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

idx = 0
for (root, dirs, files) in os.walk("refs/"):
    for d in dirs:
        shutil.rmtree("refs/" + d)

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/MIQA"
patients = os.listdir(data_path)
# Trim private data
patients = [patient for patient in patients if ("MIQA" not in patient)]
np.random.shuffle(patients)

config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
labels = list(config.maps.keys())

explore_labels = labels.copy()
explore_labels.remove("Unknown")

width = 12
height = width / 2

pat_idx = 0

cdict = {
    'alpha': (
        (0.0,  1.0, 1.0),
        (1.0,  0.0, 0.0),
    ),
    'red': (
        (0.0,  0.0, 0.0),
        (1.0,  0.0, 0.0),
    ),
    'green': (
        (0.0,  0.0, 0.0),
        (1.0,  1.0, 1.0),
    ),
    'blue': (
        (0.0,  0.0, 0.0),
        (1.0,  0.0, 0.0),
    )
}
green_cmap = LinearSegmentedColormap('BlueRed1', cdict)
fig = plt.figure(figsize=(width, height), dpi=30, frameon=False)
fig.tight_layout()
plot_ct = plt.imshow(np.zeros((512, 512)), vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
plot_seg = plt.imshow(np.zeros((512, 512)), vmin=0, vmax=2, cmap=green_cmap, alpha=.5, interpolation='nearest')
title = plt.title('')
plt.xticks([], [])
plt.yticks([], [])
while ((pat_idx < len(patients))):
    try:
        patient = patients[pat_idx]
        patient_maps = load_volume(data_path + "/" + patient, labels)
        CT = np.clip(patient_maps["CT"] / 1000, -1, 1)

        available_keys = []
        for selected_key in explore_labels:
            if ((len(patient_maps[selected_key]) > 0) & (np.sum(patient_maps[selected_key]) > 0)):
                explore_labels.remove(selected_key)

                if (not os.path.isdir('refs/' + str(selected_key))):
                    os.mkdir('refs/' + str(selected_key))

                segmentation = patient_maps[selected_key]

                first = np.min(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0))
                plot_idx = 0
                for i in np.linspace(first, last, 20, dtype=int):
                    plot_ct.set_data(np.fliplr(np.flipud(np.transpose(CT[i, :, :]))))
                    plot_seg.set_data(np.fliplr(np.flipud(np.transpose(segmentation[i, :, :]))))
                    fig.savefig("refs/" + selected_key + "/cor" + str(plot_idx) + ".jpg")
                    plot_idx += 1

                first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0))
                plot_idx = 0
                for i in np.linspace(first, last, 20, dtype=int):
                    plot_ct.set_data(np.flipud(np.transpose(CT[:, i, :])))
                    plot_seg.set_data(np.flipud(np.transpose(segmentation[:, i, :])))
                    fig.savefig("refs/" + selected_key + "/sag" + str(plot_idx) + ".jpg")
                    plot_idx += 1

                first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0))
                plot_idx = 0
                for i in np.linspace(first, last, 20, dtype=int):
                    plot_ct.set_data(CT[:, :, i])
                    plot_seg.set_data(segmentation[:, :, i])
                    fig.savefig("refs/" + selected_key + "/ax" + str(plot_idx) + ".jpg")
                    plot_idx += 1

                gc.collect()
    except Exception as e:
        print(e)
    
    pat_idx += 1