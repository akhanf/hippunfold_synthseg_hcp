#adapted from Jordan DeKraker's scripts for ex vivo hippunfold synthseg 

import sys

#this is where synthseg is located inside the container
sys.path.append('/SynthSeg')

import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator
import pandas as pd

label_img_or_dir = sys.argv[1]
label_class_tsv = sys.argv[2]
n_examples = int(sys.argv[3])
generated_dir = sys.argv[4]
out_prefix = sys.argv[5]

generation_class_name_to_label = {
  'BG': 0,
  'WM': 1,
  'GM': 2,
  'CSF': 3,
  'Vessel': 4,
  'ExtraHipp': 5,
  'SRLM': 6}

df = pd.read_table(label_class_tsv)
print(df)

generation_labels=df.generation_label.to_numpy()
output_labels=df.output_label.to_numpy()
generation_classes=np.array([generation_class_name_to_label[name] for name in list(df.generation_class_name)])

print(f'generation_labels: {generation_labels}')
print(f'output_labels: {output_labels}')
print(f'generation classes: {generation_classes}')

brain_generator = BrainGenerator(label_img_or_dir,
         generation_labels=generation_labels,
         output_labels=output_labels,
         generation_classes=generation_classes,
         target_res=0.3,
         max_res_iso=2.,
         max_res_aniso=4.)

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_brain()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,f'{generated_dir}/imagesTr/{out_prefix}_{n:05d}_0000.nii.gz')
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,f'{generated_dir}/labelsTr/{out_prefix}_{n:05d}.nii.gz')
