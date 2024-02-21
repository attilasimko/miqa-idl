import sys
sys.path.append("../miqa-seg")
import os
import numpy as np
sys.path.insert(1, os.path.abspath('.'))
from utils import model_config, load_volume
import matplotlib.pyplot as plt
plt.ion()
from skimage.measure import label 

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/MIQA"
patients = os.listdir(data_path)
labels_count = 0
config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
labels = list(config.maps.keys())

for patient in patients:
    patient_maps = load_volume(data_path + "/" + patient, labels)
    for key in patient_maps.keys():
        if (key == "CT"):
            continue
        
        if (len(patient_maps[key]) > 0):
            labels_count += 1

print(f"Number of patients: {len(patients)}")
print(f"Number of labels: {labels_count}")