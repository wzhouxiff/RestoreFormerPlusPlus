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

exp_name='2023-08-17T17-44-56_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc05_seed29'
epoch=110

exp_name='2023-08-16T10-41-25_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_seed13'
epoch=19

exp_name='2023-08-24T17-15-56_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc065_seed63'
epoch=10 # 77
epoch=17

exp_name='2023-08-25T19-17-31_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc08_seed35'
epoch=13 # 82.4940
epoch=11 # 80.9992

exp_name='2023-08-26T21-47-57_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc08_lr6_seed96'
epoch=9


exp_name='2023-08-28T16-26-15_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc08_distill010202020202_seed32'
epoch=2 # 80.3320
epoch=10 # 76.1336
epoch=12 # 79.9700
epoch=14 # 79.0212
epoch=15 # 80.0507
epoch=17 # 80.0506

exp_name='2023-08-30T00-34-34_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc12_distill010202020202_seed64'
epoch=6 # 76.4387
epoch=8

exp_name='2023-08-30T15-43-24_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc20_distill010202020202_seed69'
epoch=10

exp_name='2023-08-31T10-59-30_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc08_distill010202020202_style_seed52'
epoch=2
epoch=12

exp_name='2023-09-14T11-20-25_RestoreFormer++RestoreFormer++_gpus7_lmdb_h4_fixE_lr8_seed18'
epoch=1

exp_name='2023-09-14T13-12-55_RestoreFormer++RestoreFormer++_gpus7_lmdb_h4_fixD_lr8_seed13'
epoch=1

out_root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/WebPhoto-Test'
align_test_path='/group/30042/zhouxiawang/data/FaceRestoration/WebPhoto-Test'

out_root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/celeba'
align_test_path='/group/30042/zhouxiawang/data/FaceRestoration/celeba_512_validation_lq'

out_root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/lfw'
align_test_path='/group/30042/zhouxiawang/data/FaceRestoration/lfw'



# out_root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/Child'
# align_test_path='/group/30042/public_datasets/RestoreData/faces/CelebChild/Child'

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
