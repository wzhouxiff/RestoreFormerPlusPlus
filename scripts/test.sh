
exp_name='RestoreFormer'
exp_name='RestoreFormerPlusPlus'

root_path='experiments'
out_root_path='results'
align_test_path='data/test'
tag='test'

outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir -m 777 $outdir
fi

CUDA_VISIBLE_DEVICES=7 python -u scripts/test.py \
--outdir $outdir \
-r $root_path'/'$exp_name'/last.ckpt' \
-c 'configs/'$exp_name'.yaml' \
--test_path $align_test_path \
--aligned

