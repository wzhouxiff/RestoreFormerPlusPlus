
### Journal ###
root='results/'
out_root='results/metrics'

root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/WebPhoto-Test'
# root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/Child'
# root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/lfw'
CelebAHQ_GT=''

# root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/celeba'
# CelebAHQ_GT='/group/30042/zhouxiawang/data/FaceRestoration/celeba_512_validation'

out_root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/metrics'

test_name='RestoreFormer++'

test_name='2022-11-16T14-55-40_RestoreFormer++RestoreFormer++_lmdb_gpus8_h4_E62_ROHQD105_seed32'
test_name='2022-11-21T10-49-14_RestoreFormer++RestoreFormer++_lmdb_gpus8_h4_E62_ROHQD105_En105_lr7_seed44'
test_name='2023-08-17T17-44-56_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc05_seed29'
test_name='2023-08-24T17-15-56_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc065_seed63'
test_name='2023-08-25T19-17-31_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc08_seed35'
test_name='2023-08-26T21-47-57_RestoreFormer++_cfRestoreFormer++_cf_gpus7_lmdb_h4_ROHQD105_disc08_lr6_seed96'
test_name='2023-08-28T16-26-15_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc08_distill010202020202_seed32'
test_name='2023-08-30T00-34-34_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc12_distill010202020202_seed64'
test_name='2023-08-30T15-43-24_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc20_distill010202020202_seed69'
test_name='2023-08-31T10-59-30_2023082801_RF++_cf2023082801_RF++_cf_gpus7_lmdb_h4_ROHQD105_disc08_distill010202020202_style_seed52'
test_name='2023-09-14T13-12-55_RestoreFormer++RestoreFormer++_gpus7_lmdb_h4_fixD_lr8_seed13'

tag='test'



test_image=$test_name'_'$tag'/restored_faces'
out_name=$test_name
need_post=1




# FID
python -u scripts/metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512_color_gray008_shift02.pth' \
--save_name $out_root'/'$out_name'_fid_gray008shift02.txt' \

# --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
# --save_name $out_root'/'$out_name'_fid.txt' \

# --fid_stats 'experiments/pretrained_models/inception_FFHQ_512_color_gray008_shift02.pth' \
# --save_name $out_root'/'$out_name'_fid_gray008shift02.txt' \

if [ -d $CelebAHQ_GT ]
then
    # PSRN SSIM LPIPS
    python -u scripts/metrics/cal_psnr_ssim.py \
    $root'/'$test_image \
    --gt_folder $CelebAHQ_GT \
    --save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
    --need_post $need_post \

    # # # PSRN SSIM LPIPS
    python -u scripts/metrics/cal_identity_distance.py  \
    $root'/'$test_image \
    --gt_folder $CelebAHQ_GT \
    --save_name $out_root'/'$out_name'_id.txt' \
    --need_post $need_post
else
    echo 'The path of GT does not exist'
fi