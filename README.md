# FMPN-FER



Official PyTorch Implementation for VCIP 2019 Oral Paper **Facial Motion Prior Networks for Facial Expression Recognition**





```
python >= 3.6
pytorch >= 0.4.1
visdom == 0.1.8.9
```


## Pretrain the Facial-Motion Mask Generator (FMG)

``` sh
python main.py --mode train --data_root datasets/CKPlus --train_csv train_ids_0.csv --print_losses_freq 4 --use_data_augment --visdom_env res_baseline_ckp_0 --niter 150 --niter_decay 150 --lucky_seed 1218 --gpu_ids 0 --model res_baseline --solver resface --img_nc 1 --sample_img_freq 2 
```

## Jointly Training 

Load previous trained weight for the FMG, and ImageNet pretrained weight for the Classifer Net.

```sh
python main.py --mode train --data_root datasets/CKPlus --train_csv train_ids_0.csv --print_losses_freq 4 --use_data_augment --visdom_env res_cls_ckp_0 --lucky_seed 1218 --niter 100 --niter_decay 100 --gpu_ids 0 --model res_cls --solver res_cls --lambda_resface 0.1 --batch_size 16 --backend_pretrain --load_model_dir ckpts/CKPlus/res_baseline/fold_0/190117_165651 --load_epoch 300
```

## Testing 

``` sh
python main.py --mode test --data_root datasets/CKPlus --test_csv test_ids_0.csv --gpu_ids 0 --model res_cls --solver res_cls --batch_size 4 --load_model_dir ckpts/CKPlus/res_cls/fold_0/190118_170050 --load_epoch 200 
```

