# # ### Good
exp_name='RestoreFormer'
exp_name='RestoreFormer++'

root_path='experiments'
out_root_path='results'
align_test_path='data/test'
tag='test'


root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/release/logs'
# exp_name='2022-11-16T14-55-40_RestoreFormer++RestoreFormer++_lmdb_gpus8_h4_E62_ROHQD105_seed32'
# epoch='105'

exp_name='2022-11-21T10-49-14_RestoreFormer++RestoreFormer++_lmdb_gpus8_h4_E62_ROHQD105_En105_lr7_seed44'
epoch=63
epoch=109

out_root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/WebPhoto-Test'
align_test_path='/group/30042/zhouxiawang/data/FaceRestoration/WebPhoto-Test'

outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir -m 777 $outdir
fi

CUDA_VISIBLE_DEVICES=7 python -u scripts/test.py \
--outdir $outdir \
-r $root_path'/'$exp_name'/checkpoints/last.ckpt.'$epoch \
-c 'configs/RestoreFormer++.yaml' \
--test_path $align_test_path \
--aligned

