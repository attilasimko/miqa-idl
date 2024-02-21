import sys
sys.path.append("../miqa-seg")
import os
import numpy as np
sys.path.insert(1, os.path.abspath('.'))
from utils import model_config, load_volume
from model import OurNet
import shutil
import re
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

def custom_sort(string):
    
    match = re.match(r'([^\d]+)(\d+)', string)
    if match:
        non_numeric, numeric = match.groups()
        return (non_numeric, int(numeric))
    else:
        return (string, 0)

idx = 0
while True:
    if (os.path.isdir("private/" + str(idx))):
        shutil.rmtree("private/" + str(idx))
        idx += 1
    else:
        break
while True:
    if (os.path.isdir("public/" + str(idx))):
        shutil.rmtree("public/" + str(idx))
        idx += 1
    else:
        break

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/MIQA"
patients = os.listdir(data_path)
# Trim private data
# patients = [patient for patient in patients if ("Pelvis" not in patient) & ("Breast" not in patient)]
np.random.shuffle(patients)

config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
# patients = np.array(patients)[np.random.choice(len(patients), size=N, replace=False)]
# patients = sorted(patients)[800:800+N]
model_name = "grumpy_couch_4240_hero"
model_path = "/home/attilasimko/Documents/out/miqa/" + model_name + ".h5"
model = "unet"
labels = list(config.maps.keys())

explore_labels = os.listdir("refs/")
explore_labels = sorted(explore_labels, key=custom_sort)
# explore_labels = ["Cochlea_L", "Cochlea_R", "Pituitary", "Arytenoid", "Chiasm", "OpticNerve_L", "OpticNerve_R", "Submandibular_L", "Submandibular_R"]

# Build model
network = OurNet()
if (model == "nnformer"):
    num_filters = 48
    base_model = network.build_nnformer(len(config.labels), 1, num_filters, 0.0, 0.0, 0.0)
elif (model == "unet"):
    num_filters = 48
    base_model = network.build_unet(len(config.labels), 1, num_filters, 0.0, config.epsilon, 0.0)
else:
    raise ValueError("Model not recognized.")
base_model.load_weights(model_path)
base_model.compile(loss="mse")

width = 12
height = width / 2
text_private = "patient,label,model,coronal,sagittal,axial\n"
text_public = "patient,label,model,coronal,sagittal,axial\n"

N = 5
num_labels = 5
min_voxels = 10
pat_idx = 0
saved_idx_private = 0
saved_idx_public = 0

fig = plt.figure(figsize=(width, height), dpi=30, frameon=False)
fig.tight_layout()
plot_ct = plt.imshow(np.zeros((512, 512)), vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
plot_seg = plt.imshow(np.zeros((512, 512)), vmin=0, vmax=2, cmap="gist_heat", alpha=.5, interpolation='nearest')
title = plt.title('')
plt.xticks([], [])
plt.yticks([], [])

while (((saved_idx_private < N) | (saved_idx_public < N)) & (pat_idx < len(patients))):
    try:
        patient = patients[pat_idx]
        if (("Pelvis" in patient) | ("Breast" in patient)):
            save_path = "private/"
            is_private = True
        else:
            save_path = "public/"
            is_private = False
    
        saved_idx = saved_idx_private if is_private else saved_idx_public
        if (saved_idx >= N):
            pat_idx += 1
            continue

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
            saved_idx = saved_idx_private if is_private else saved_idx_public
            if (saved_idx < N):
                if (not os.path.isdir(save_path + str(saved_idx))):
                    os.mkdir(save_path + str(saved_idx))
                line = (patient + ',')
                line += (selected_key + ',')
                line += (model_name + ',')

                segmentation = patient_pred == config.mapping[selected_key]

                first = np.min(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(1, 2)) != 0)) + 1
                plot_idx = 0
                for i in range(first, last):
                    plot_ct.set_data(np.fliplr(np.flipud(np.transpose(CT[i, :, :]))))
                    plot_seg.set_data(np.fliplr(np.flipud(np.transpose(segmentation[i, :, :]))))
                    fig.savefig(save_path + str(saved_idx) + "/cor" + str(plot_idx) + ".png")
                    plot_idx += 1
                line += (str(plot_idx) + ',')

                first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 2)) != 0)) + 1
                plot_idx = 0
                for i in range(first, last):
                    plot_ct.set_data(np.flipud(np.transpose(CT[:, i, :])))
                    plot_seg.set_data(np.flipud(np.transpose(segmentation[:, i, :])))
                    fig.savefig(save_path + str(saved_idx) + "/sag" + str(plot_idx) + ".png")
                    plot_idx += 1
                line += (str(plot_idx) + ',')

                first = np.min(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0))
                last = np.max(np.argwhere(np.sum(segmentation, axis=(0, 1)) != 0)) + 1
                plot_idx = 0
                for i in range(first, last):
                    plot_ct.set_data(CT[:, :, i])
                    plot_seg.set_data(segmentation[:, :, i])
                    fig.savefig(save_path + str(saved_idx) + "/ax" + str(plot_idx) + ".png")
                    plot_idx += 1
                line += (str(plot_idx) + '\n')

                if (is_private):
                    text_private += line
                    saved_idx_private += 1
                else:
                    text_public += line
                    saved_idx_public += 1
                gc.collect()
    except Exception as e:
        print(e)
    
    pat_idx += 1

with open('patientlog_private.txt', 'w') as f:
    f.write(text_private)
with open('patientlog_public.txt', 'w') as f:
    f.write(text_public)