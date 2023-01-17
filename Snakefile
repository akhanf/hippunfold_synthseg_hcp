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

hemis=['L','R']

rule all_synthmri:
    input: 
        generated_tar=expand(join(root,'{subject}_hemi-{hemi}_n-{n_examples}_generated.tar'),subject=subjects,hemi=hemis,n_examples=config['n_examples']),



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
    """set everything >1001 to label 3 (ie all cortical GM to same label)"""
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
    """ merge by adding 1000 to labels in the hippocampus (1 through 8 now 1001 to 1008) """
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
        out_prefix='{subject}_{hemi}'
    threads: 32
    output:
        generated_tar=join(root,'{subject}_hemi-{hemi}_n-{n_examples}_generated.tar')
    resources:
        gpus=1,
        mem_mb=32000,
        time=10
    group: 
        'synth'
    shell: 
        'singularity exec --nv {input.container} python {input.script} ' #args below
        ' {input.label_img_or_dir}'
        ' {input.label_class_tsv}' 
        ' {params.n_examples} '
        ' {resources.tmpdir}/{params.generated_dir}/imgs '
        ' {params.out_prefix} '
        ' && ' #tar it up afterwards
        ' tar -cvf {output.generated_tar} -C {resources.tmpdir}/{params.generated_dir} imgs '

