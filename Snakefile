#use hcp hippunfold training data with synthseg

# we want to merge hippunfold manual labels and freesurfer labels

# 1. find affine xfm from hcp brain to corobl (since we don't have this anymore)
# 2. transform freesurfer aseg corobl
# 3. merge the labels (labelmerge?)

configfile: 'config.yml'

from os.path import join

root='results'
tmp_dir='/tmp/akhanf'

with open("subjects.txt",'r') as subjtext:
    subjects = subjtext.readlines()

subjects=[subj.strip() for subj in subjects]
#subjects=subjects[:3] #first 10 subjects for quick test

#do 80/20 split of training/test
import os
import random
random.seed(0)
num_training = int(0.8 * len(subjects))
shuffle_subjects = random.sample(subjects,k=len(subjects))
training_subjects = shuffle_subjects[:num_training]
testing_subjects = shuffle_subjects[num_training:]
print(f'number of training subjects: {len(training_subjects)}')
print(f'number of test subjects: {len(testing_subjects)}')



hemis=['L','R']

rule all_synthmri:
    input: 
        generated_tar=expand(join(root,'{subject}_hemi-{hemi}_n-{n_examples}_generated.tar'),subject=subjects,hemi=hemis,n_examples=config['n_examples']),

rule all_preproc:
    input:
        preprocessed_tar = expand('nnunet_data/preprocessed_{task}.tar',task=config['task'])

rule all_train:
    input:
        training_done = expand('nnunet_data/training_done_fold-{fold}.{task}',fold=range(5),task=config['task'])

rule all_merged:
    input: 
        merged=expand(join(root,'{subject}_hemi-{hemi}_mergedseg.nii.gz'),subject=subjects,hemi=hemis),

rule unzip_t1:
    input:
        zipfile=config['training_zip']
    params:
        in_file='{subject}_hemi-{hemi}_T1w.nii.gz'
    output:
        join(root,'{subject}_hemi-{hemi}_T1w.nii.gz')
    group: 'subj'
    shell:
        'unzip {input.zipfile} -d {root} {params.in_file}'

rule unzip_seg:
    input:
        zipfile=config['training_zip']
    params:
        in_file='{subject}_hemi-{hemi}_lbl.nii.gz'
    output:
        join(root,'{subject}_hemi-{hemi}_lbl.nii.gz')
    group: 'subj'
    shell:
        'unzip {input.zipfile} -d {root} {params.in_file}'


rule reg_brain_to_corobl:
    input:
        fixed=join(root,'{subject}_hemi-{hemi}_T1w.nii.gz'),
        moving=config['brain_t1'],
        init_rigid_xfm='resources/init_rigid_xfm.txt'
    output:
        rigid_xfm=join(root,'{subject}_hemi-{hemi}_rigid.txt'),
        affine_xfm=join(root,'{subject}_hemi-{hemi}_affine.txt'),
    threads: 8
    group: 'subj'
    shell:
        'greedy -threads {threads} -d 3 -i {input.fixed} {input.moving} -o {output.rigid_xfm} -m NCC 5x5x5 -a -dof 6 -ia {input.init_rigid_xfm} && '
        ' greedy -threads {threads} -d 3 -i {input.fixed} {input.moving} -o {output.affine_xfm} -m NCC 5x5x5 -a -dof 12 -ia {output.rigid_xfm}'

rule reslice_brain_to_corobl:
    input:
        fixed=join(root,'{subject}_hemi-{hemi}_T1w.nii.gz'),
        moving=config['brain_t1'],
        aparc_aseg=config['aparc_aseg'],
        xfm=join(root,'{subject}_hemi-{hemi}_affine.txt'),
    params:
        label_smoothing='0.6mm'
    output:
        reg_t1=join(root,'{subject}_hemi-{hemi}_regbrain.nii.gz'),
        aparc_aseg=join(root,'{subject}_hemi-{hemi}_aparc+aseg.nii.gz'),
    threads: 8
    group: 'subj'
    shell:
        'greedy -threads {threads} -d 3 -r {input.xfm} -rf {input.fixed} '
        ' -ri LINEAR -rm {input.moving} {output.reg_t1} '
        ' -ri LABEL {params.label_smoothing} -rm {input.aparc_aseg} {output.aparc_aseg} '

rule smooth_hipp_lbls:
    input:
        seg=join(root,'{subject}_hemi-{hemi}_lbl.nii.gz'),
        xfm='resources/identity_xfm.txt'
    params:
        label_smoothing='0.3mm'
    output:
        seg=join(root,'{subject}_hemi-{hemi}_hippunfold.nii.gz'),
    threads: 8
    group: 'subj'
    shell:
        'greedy -threads {threads} -d 3 -r {input.xfm} -rf {input.seg} '
        ' -ri LABEL {params.label_smoothing}  -rm {input.seg} {output.seg} '

rule clean_aparc:
    #set everything >1001 to label 3 (ie all cortical GM to same label)
    input:
        aparc_aseg=join(root,'{subject}_hemi-{hemi}_aparc+aseg.nii.gz'),
    output:
        clean_aseg=join(root,'{subject}_hemi-{hemi}_cleanaseg.nii.gz'),
    group: 'subj'
    shell:
        'c3d {input.aparc_aseg} -threshold 1000 Inf 3 0 -popas GM '
        ' {input.aparc_aseg} -threshold 1000 Inf 0 1  -popas NONGM '
        ' {input.aparc_aseg} -push NONGM -multiply -popas APARC_NO_GM '
        ' -push GM -push APARC_NO_GM -add -o {output.clean_aseg} '

 

rule merge_labels:
    #merge by adding 1000 to labels in the hippocampus (1 through 8 now 1001 to 1008) 
    input:
        hippunfold=join(root,'{subject}_hemi-{hemi}_hippunfold.nii.gz'),
        aseg=join(root,'{subject}_hemi-{hemi}_cleanaseg.nii.gz'),
    output:
        merged=join(root,'{subject}_hemi-{hemi}_mergedseg.nii.gz')
    group: 'subj'
    shell:
        'c3d {input.hippunfold} -binarize -popas BINARY_HIPP  ' 
        ' -push BINARY_HIPP -scale 1000 {input.hippunfold} -add -popas RESCALED_HIPP '
        ' -push BINARY_HIPP -replace 1 0 0 1 -popas BINARY_NONHIPP '
        ' {input.aseg} -push BINARY_NONHIPP -multiply -popas APARC_NO_HIPP '
        ' -push RESCALED_HIPP -push APARC_NO_HIPP -add -popas MERGED '
        ' -push MERGED -o {output.merged}'
   


rule generate_synth_mri:
    input:
        label_img_or_dir=join(root,'{subject}_hemi-{hemi}_mergedseg.nii.gz'),
        label_class_tsv='resources/synthseg_classes.tsv',
        container='/project/6050199/akhanf/singularity/bids-apps/synthseg_v2.0.sif',
        script='scripts/generate.py'
    params:
        n_examples='{n_examples}',
        generated_dir='{subject}_hemi-{hemi}_n-{n_examples}_generated',
        out_prefix='{subject}_{hemi}',
        task=config['task']
    threads: 32
    output:
        generated_tar=join(root,'{subject}_hemi-{hemi}_n-{n_examples}_generated.tar')
    resources:
        #gpus=1,
        mem_mb=32000,
        time=10
    group: 
        'synth'
    shell: 
        #'singularity exec --nv {input.container} python {input.script} ' #args below
        'singularity exec {input.container} python {input.script} ' #args below
        ' {input.label_img_or_dir}'
        ' {input.label_class_tsv}' 
        ' {params.n_examples} '
        ' {resources.tmpdir}/{params.generated_dir} '
        ' {params.out_prefix} '
        ' && ' #tar it up afterwards
        ' tar -cvf {output.generated_tar} -C {resources.tmpdir}/{params.generated_dir} imagesTr labelsTr '


localrules: create_dataset_json, extract_rawdata_tars

rule extract_rawdata_tars:
    input:
        generated_tars=expand(join(root,'{subject}_hemi-{hemi}_n-{n_examples}_generated.tar'),subject=training_subjects,hemi=hemis,n_examples=config['n_examples']),
        dataset_json = 'nnunet_data/{task}_dataset.json'
    output:
        raw_data_dir = directory('nnunet_data/raw_data/nnUNet_raw_data/{task}')
    shell:
        'mkdir -p {output.raw_data_dir} && '
        'cp -v {input.dataset_json} {output.raw_data_dir}/dataset.json && ' #copy json
        'for tar in {input.generated_tars}; '
        'do'
        '  tar -xvf $tar -C {output.raw_data_dir} imagesTr labelsTr; '
        'done'


rule create_dataset_json:
    input: 
        template_json = 'resources/nnunet_template.json'
    params:
        training_lbls =  expand('nnunet_data/raw_data/{task}/labelsTr/{subject}_{hemi}_{sample:05d}.nii.gz',subject=training_subjects,hemi=hemis,sample=range(config['n_examples']),allow_missing=True),
        training_imgs =  expand('nnunet_data/raw_data/{task}/imagesTr/{subject}_{hemi}_{sample:05d}.nii.gz',subject=training_subjects,hemi=hemis,sample=range(config['n_examples']),allow_missing=True)
    output: 
        dataset_json = 'nnunet_data/{task}_dataset.json'
    script: 'scripts/create_nnunet_json.py' 

def get_nnunet_env(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env'].items()])
 
def get_nnunet_env_tmp(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env_tmp'].items()])
 
rule plan_preprocess:
    input: 
        raw_data = 'nnunet_data/raw_data/nnUNet_raw_data/{task}'
    params:
        nnunet_env_cmd = get_nnunet_env_tmp,
        task_num = lambda wildcards: re.search('Task([0-9]+)\w*',wildcards.task).group(1),
    output: 
        preprocessed_tar = 'nnunet_data/preprocessed_{task}.tar'
    group: 'preproc'
    threads: 16
    resources:
        mem_mb = 32000,
        time = 1440,
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNet_plan_and_preprocess  -t {params.task_num} --verify_dataset_integrity && '
        'tar -cvf {output.preprocessed_tar} -C $SLURM_TMPDIR preprocessed'

  
def get_checkpoint_opt(wildcards, output): #NOTE with shell outputs are deleted on calling this rule :(
    if os.path.exists('nnunet_data/trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_latest.model'.format(fold=wildcards.fold, task=wildcards.task,arch=config['architecture'],trainer=config['trainer'])):
        return '--continue_training'
    else:
        return '' 
     
rule train_fold:
    input:
        preprocessed_tar = 'nnunet_data/preprocessed_{task}.tar'
    params:
        nnunet_env_cmd = get_nnunet_env_tmp,
        mkdir_tmp = expand('mkdir -p {env_tmp}', env_tmp=config['nnunet_env_tmp']['nnUNet_preprocessed']),
        rsync_to_tmp = lambda wildcards: expand('rsync -av {env}/{t}/ {env_tmp}/{t}', env=config['nnunet_env']['nnUNet_preprocessed'], env_tmp=config['nnunet_env_tmp']['nnUNet_preprocessed'], t=wildcards.task),
        #add --continue_training option if a checkpoint exists
        checkpoint_opt = get_checkpoint_opt,
        arch=config['architecture'],
        trainer=config['trainer']
    output:
        training_done = touch('nnunet_data/training_done_fold-{fold}.{task}')
#        model_dir = directory('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}'),
#        final_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_final_checkpoint.model',
#        latest_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_latest.model',
#        best_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_best.model'
    threads: 32
    resources:
        gpus = 1,
        mem_mb = 64000,
        time = 4320,
    group: 'train'
    shell:
        '{params.nnunet_env_cmd} && '
        'tar -xvf {input.preprocessed_tar} -C $SLURM_TMPDIR && ' 
        'nnUNet_train {params.checkpoint_opt} {params.arch} {params.trainer} {wildcards.task} {wildcards.fold}'


