import sys

sys.path.append('/SynthSeg')
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator
import pandas as pd

print('start of script')
df = pd.read_table(snakemake.input.label_class_tsv)
print(df)

generation_labels=df.generation_label.to_numpy()
output_labels=df.output_label.to_numpy()
generation_classes=np.array([snakemake.config['generation_class_name_to_label'][name] for name in list(df.generation_class_name)])

print(f'generation_labels: {generation_labels}')
print(f'output_labels: {output_labels}')
print(f'generation classes: {generation_classes}')

brain_generator = BrainGenerator(snakemake.input.label_dir,
         generation_labels=generation_labels,
         output_labels=output_labels,
         generation_classes=generation_classes,
         target_res=0.3,
         max_res_iso=2.,
         max_res_aniso=4.)


for n in range(snakemake.params.n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_brain()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(snakemake.output.generated_dir, 'image_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(snakemake.output.generated_dir, 'labels_%s.nii.gz' % n))
