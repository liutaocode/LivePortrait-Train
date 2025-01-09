python train.py \
    --batch_size 8 \
    --val_batch_size 4 \
    --lr_g 1e-4 \
    --lr_d 4e-4 \
    --exp_name "exp_wingloss" \
    --exp_dir "./exps/exps3/" \
    --cache_dir "./assets/db_cache/" \
    --vgg_loss_weight 1.0 \
    --gan_loss_weight 0.1 \
    --prior_loss_weight 1.0 \
    --deformation_loss_weight 1.0 \
    --headpose_loss_weight 1.0 \
    --equivariance_loss_weight 1.0 \
    --every_n_epochs 1 \
    --recon_loss_weight 10.0 \
    --pretrained_mode 2 \
    --checkpoint_path "" \
    --max_epochs 1000 \
    --debug_mode False \
    --wandb_mode True \
    --clip_grad_norm 1.0 \
    --wing_loss_omega 10 \
    --wing_loss_epsilon 2 \
    --landmark_selected_index "36,39,37,42,45,43,48,54,51,57"
