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
while True:
    if (os.path.isdir(str(idx))):
        shutil.rmtree(str(idx))
        idx += 1
    else:
        break

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/MIQA"
patients = os.listdir(data_path)
# Trim private data
patients = [patient for patient in patients if ("Pelvis" not in patient) & ("Breast" not in patient)]
np.random.shuffle(patients)

config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
labels = list(config.maps.keys())

explore_labels = labels.copy()
explore_labels.remove("Unknown")

width = 12
height = width / 2
text = "patient,label,model,coronal,sagittal,axial\n"

N = 500
num_labels = 10
min_voxels = 50
pat_idx = 0
saved_idx = 0

fig = plt.figure(figsize=(width, height), dpi=30, frameon=False)
fig.tight_layout()
plot_ct = plt.imshow(np.zeros((512, 512)), vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
plot_seg = plt.imshow(np.zeros((512, 512)), vmin=0, vmax=2, cmap="gist_heat", alpha=.5, interpolation='nearest')
title = plt.title('')
plt.xticks([], [])
plt.yticks([], [])

while ((pat_idx < len(patients))):
    try:
        patient = patients[pat_idx]
        patient_maps = load_volume(data_path + "/" + patient, labels)
        CT = np.clip(patient_maps["CT"] / 1000, -1, 1)
        patient_pred = np.zeros((CT.shape), np.float32)
        new_pred = np.zeros((CT.shape), np.int32)
        for i in range(CT.shape[2]):
            patient_pred[:, :, i] = np.argmax(base_model.predict(np.moveaxis(CT[:, :, i:i+1, np.newaxis], 2, 0), verbose=0)[0][0, :, :, :], -1)
    
        for idx in range(len(labels)):
            largest_seg = getLargestCC(patient_pred == idx)
            if (np.sum(largest_seg) > np.sum(patient_pred == idx) * 0.9):
                new_pred += largest_seg * config.mapping[labels[idx]]
        patient_pred = new_pred

        available_keys = []
        for key in explore_labels:
            if (len(patient_maps[key]) > 0):
                continue

            if (np.sum(patient_pred == config.mapping[key]) > min_voxels):
                available_keys.append(key)

        np.random.shuffle(available_keys)
        available_keys = available_keys[:num_labels]
        for selected_key in available_keys:
            if (saved_idx < N):
                if (not os.path.isdir(str(saved_idx))):
                    os.mkdir(str(saved_idx))
                text += (patient + ',')
                text += (selected_key + ',')

                segmentation = patient_maps[selected_key]

                first = np.min(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0)) + 1
                plot_idx = 0
                for i in range(first, last):
                    plot_ct.set_data(np.fliplr(np.flipud(np.transpose(CT[i, :, :]))))
                    plot_seg.set_data(np.fliplr(np.flipud(np.transpose(segmentation[i, :, :]))))
                    fig.savefig("refs/" + selected_key + "/cor" + str(plot_idx) + ".png")
                    plot_idx += 1
                text += (str(plot_idx) + ',')

                first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0)) + 1
                plot_idx = 0
                for i in range(first, last):
                    plot_ct.set_data(np.flipud(np.transpose(CT[:, i, :])))
                    plot_seg.set_data(np.flipud(np.transpose(segmentation[:, i, :])))
                    fig.savefig("refs/" + selected_key + "/sag" + str(plot_idx) + ".png")
                    plot_idx += 1
                text += (str(plot_idx) + ',')

                first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0)) + 1
                plot_idx = 0
                for i in range(first, last):
                    plot_ct.set_data(CT[:, :, i])
                    plot_seg.set_data(segmentation[:, :, i])
                    fig.savefig("refs/" + selected_key + "/ax" + str(plot_idx) + ".png")
                    plot_idx += 1
                text += (str(plot_idx) + '\n')

                saved_idx += 1
                gc.collect()
    except Exception as e:
        print(e)
    
    pat_idx += 1

with open('reflog.txt', 'w') as f:
    f.write(text)