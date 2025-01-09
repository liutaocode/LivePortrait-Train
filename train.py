import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L
import yaml
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.modules.spade_generator import SPADEDecoder
from src.modules.warping_network import WarpingNetwork
from src.modules.motion_extractor import MotionExtractor
from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
from src.losses import KeypointPriorLoss, EquivarianceLoss, WingLoss, HeadPoseLoss, DeformationPriorLoss
from src.datasets import CustomDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.vgg19 import VGGLoss
from src.discriminator import Discriminator
import torch.nn.functional as F
import argparse
import time
from pytorch_lightning import seed_everything

class LitAutoEncoder(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.automatic_optimization = False # for GAN training

        models_config = 'src/config/models.yaml'

        model_config = yaml.load(open(models_config, 'r'), Loader=yaml.SafeLoader)

        self.appearance_feature_extractor = AppearanceFeatureExtractor(**model_config['model_params']['appearance_feature_extractor_params'])
        self.motion_extractor = MotionExtractor(**model_config['model_params']['motion_extractor_params'])
        self.warping_module = WarpingNetwork(**model_config['model_params']['warping_module_params'])
        self.spade_generator = SPADEDecoder(**model_config['model_params']['spade_generator_params'])

        self.dis_gan = Discriminator()

        self._load_pretrained_weights()
        # losses
        self.keypoint_prior_loss = KeypointPriorLoss()

        self.deformation_prior_loss = DeformationPriorLoss()

        self.equivariance_loss = EquivarianceLoss(
            sigma_affine=0.05,
            sigma_tps=0.005,
            points_tps=5
        )

        self.headpose_loss = HeadPoseLoss()

        self.vgg_loss = VGGLoss()

        self.wing_loss = WingLoss(omega=self.args.wing_loss_omega, epsilon=self.args.wing_loss_epsilon)

        self.num_of_selected_landmarks = len(self.args.landmark_selected_index)
        print('self.num_of_selected_landmarks for wing loss', self.num_of_selected_landmarks)

        print(model_config)

    def _load_pretrained_weights(self):
        """Handle different pretrained model loading modes"""
        # you can download the pretrained weights from the official LivePortrait repo
        checkpoint_paths = {
            'F': 'pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth',
            'M': 'pretrained_weights/liveportrait/base_models/motion_extractor.pth',
            'G': 'pretrained_weights/liveportrait/base_models/spade_generator.pth',
            'W': 'pretrained_weights/liveportrait/base_models/warping_module.pth'
        }
        mode_handlers = {
            0: self._train_from_scratch,
            1: lambda: None,  # Resume training handled by Lightning trainer
            2: lambda: self._load_partial_model(self.args.checkpoint_path),
            3: lambda: self._load_official_weights(checkpoint_paths)
        }

        if self.args.pretrained_mode not in mode_handlers:
            raise ValueError(f"Invalid pretrained_mode: {self.args.pretrained_mode}")

        mode_handlers[self.args.pretrained_mode]()

    def _train_from_scratch(self):
        pass

    def _load_official_weights(self, checkpoint_paths):
        if os.path.exists(checkpoint_paths['F']):
            self.appearance_feature_extractor.load_state_dict(torch.load(checkpoint_paths['F']))
        else:
            print(f"Checkpoint {checkpoint_paths['F']} does not exist.")

        if os.path.exists(checkpoint_paths['M']):
            self.motion_extractor.load_state_dict(torch.load(checkpoint_paths['M']))
        else:
            print(f"Checkpoint {checkpoint_paths['M']} does not exist.")

        if os.path.exists(checkpoint_paths['W']):
            self.warping_module.load_state_dict(torch.load(checkpoint_paths['W']))
        else:
            print(f"Checkpoint {checkpoint_paths['W']} does not exist.")

        if os.path.exists(checkpoint_paths['G']):
            self.spade_generator.load_state_dict(torch.load(checkpoint_paths['G']))
        else:
            print(f"Checkpoint {checkpoint_paths['G']} does not exist.")

    def _load_partial_model(self, pretrained_model_path):
        selfState = self.state_dict()

        print(f'Loading pretrained model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Filter out unnecessary keys
        filtered_state_dict = {}
        for name, param in state_dict.items():
            # Remove 'model.' prefix if it exists
            if name.startswith('model.'):
                name = name[6:]  # Remove 'model.' prefix

            # Check if parameter exists in current model
            if name in selfState:
                if selfState[name].shape == param.shape:
                    filtered_state_dict[name] = param
                else:
                    print(f"Skipping {name} due to shape mismatch: {selfState[name].shape} vs {param.shape}")
            else:
                print(f"Skipping {name} as it's not in current model")

        # Load the filtered state dict
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)} parameters")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def process_kp(self, kp_info):
        bs = kp_info['kp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
        kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
        kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
        kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        kp = kp_info['kp']
        scale = kp_info['scale'].unsqueeze(-1)
        R = get_rotation_matrix(kp_info['pitch'], kp_info['yaw'], kp_info['roll'])
        exp = kp_info['exp']
        t = kp_info['t'].unsqueeze(1)

        return kp, scale, R, exp, t

    def training_step(self, batch, batch_idx):

        optimizer_g, optimizer_d = self.optimizers()
        optimizer_g.zero_grad()

        source_img = batch['source_img']
        target_img = batch['target_img']
        target_img_512 = batch['target_img_512']
        target_ypr = batch['target_ypr']
        target_lmd = batch['target_lmd']

        f_s = self.appearance_feature_extractor(source_img)
        x_s_info = self.motion_extractor(source_img)

        x_s_kp, x_s_scale, x_s_R, x_s_exp, x_s_t = self.process_kp(x_s_info)

        x_s_full = x_s_scale * ( x_s_kp @ x_s_R + x_s_exp) + x_s_t

        x_t_info = self.motion_extractor(target_img)

        x_t_kp, x_t_scale, x_t_R, x_t_exp, x_t_t = self.process_kp(x_t_info)

        x_d_full = x_t_scale * (x_t_kp @ x_t_R + x_t_exp) + x_t_t


        ret_dct = self.warping_module(f_s, kp_source=x_s_full, kp_driving=x_d_full)


        output_result = self.spade_generator(feature=ret_dct['out'])

        l_recon = torch.nn.functional.mse_loss(output_result, target_img_512)
        l_e = self.equivariance_loss(target_img, x_t_kp, self.motion_extractor)

        l_prior = self.keypoint_prior_loss(x_d_full)
        l_deformation = self.deformation_prior_loss(x_t_exp)

        yaw_real, pitch_real, roll_real = target_ypr[:,0], target_ypr[:,1], target_ypr[:,2]
        l_headpose = self.headpose_loss(x_t_info['yaw'], x_t_info['pitch'], x_t_info['roll'], yaw_real, pitch_real, roll_real)

        l_vgg = self.vgg_loss(output_result, target_img_512)


        l_wing = self.wing_loss(x_d_full[:, :self.num_of_selected_landmarks, :2], target_lmd)

        img_recon_pred = self.dis_gan(output_result * 2 - 1)
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        # Add gradient clipping for stable training
        if self.args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad_norm)

        # Add loss scaling to prevent any single loss from dominating
        loss_total = (
            self.args.recon_loss_weight * l_recon +
            self.args.vgg_loss_weight * l_vgg +
            self.args.equivariance_loss_weight * l_e +
            self.args.prior_loss_weight * l_prior +
            self.args.deformation_loss_weight * l_deformation +
            self.args.headpose_loss_weight * l_headpose +
            self.args.gan_loss_weight * gan_g_loss +
            self.args.wing_loss_weight * l_wing
        )

        self.log("train_loss", loss_total, on_step=True, prog_bar=True)
        self.log("recon_loss", l_recon, on_step=True)
        self.log("prior_loss", l_prior, on_step=True)
        self.log("equivariance_loss", l_e, on_step=True)
        self.log("deformation_loss", l_deformation, on_step=True)
        self.log("headpose_loss", l_headpose, on_step=True)
        self.log("vgg_loss", l_vgg, on_step=True)
        self.log("gan_generator_loss", gan_g_loss, on_step=True)
        self.log("wing_loss", l_wing, on_step=True)

        self.manual_backward(loss_total)
        optimizer_g.step()

        optimizer_d.zero_grad()

        real_img_pred = self.dis_gan(target_img_512 * 2 - 1)
        recon_img_pred = self.dis_gan(output_result.detach() * 2 - 1)
        gan_d_loss = self.args.gan_loss_weight * self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        self.manual_backward(gan_d_loss)
        self.log("gan_discriminator_loss", gan_d_loss, on_step=True)

        optimizer_d.step()


        return loss_total

    def configure_optimizers(self):
        lr_g = self.args.lr_g
        lr_d = self.args.lr_d
        betas = (0.0, 0.999)
        generator_params = list(self.appearance_feature_extractor.parameters()) + \
                        list(self.motion_extractor.parameters()) + \
                        list(self.warping_module.parameters()) + \
                        list(self.spade_generator.parameters())

        opt_g = torch.optim.Adam(
            generator_params,
            lr=lr_g,
            betas=betas
        )
        opt_d = torch.optim.Adam(
            self.dis_gan.parameters(),
            lr=lr_d,
            betas=betas
        )
        return opt_g, opt_d

    def validation_step(self, batch, batch_idx):
        # TODO
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)
    parser.add_argument("--exp_name", type=str, default="expname")
    parser.add_argument("--exp_dir", type=str, default="./exps/exps1/")
    parser.add_argument("--cache_dir", type=str, default="./assets/db_cache/")
    parser.add_argument("--vgg_loss_weight", type=float, default=0.1)
    parser.add_argument("--gan_loss_weight", type=float, default=1)
    parser.add_argument("--prior_loss_weight", type=float, default=1)
    parser.add_argument("--deformation_loss_weight", type=float, default=1)
    parser.add_argument("--headpose_loss_weight", type=float, default=1)
    parser.add_argument("--equivariance_loss_weight", type=float, default=1)
    parser.add_argument("--wing_loss_weight", type=float, default=1)
    parser.add_argument("--every_n_epochs", type=int, default=1, help="save checkpoint every n epochs")
    parser.add_argument("--recon_loss_weight", type=float, default=0, help="weight for reconstruction loss. 0 means no reconstruction loss.")
    parser.add_argument("--pretrained_mode", type=int, default=0, help="0: train from scratch, 1: resume training from lightning checkpoint, 2: partial training from lightning checkpoint, 3: train from official LivePortrait model")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--wandb_mode", type=bool, default=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--wing_loss_omega", type=float, default=10, help="omega for wing loss")
    parser.add_argument("--wing_loss_epsilon", type=float, default=2, help="epsilon for wing loss")
    parser.add_argument("--landmark_selected_index", type=str, default="36,39,37,42,45,43,48,54,51,57", help="landmark selected index for wing loss")

    args = parser.parse_args()

    seed_everything(args.seed)

    args.landmark_selected_index = [int(i) for i in args.landmark_selected_index.split(',')]

    exp_prefix_name = f"mode_{args.pretrained_mode}_time_{time.strftime('%Y%m%d%H%M%S')}"
    checkpoints_dir = f"{args.exp_dir}/{args.exp_name}/{exp_prefix_name}/checkpoints/"
    logs_dir = f"{args.exp_dir}/{args.exp_name}/{exp_prefix_name}/logs/"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save training parameters to file
    with open(os.path.join(checkpoints_dir, "train_params.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    if not args.wandb_mode or args.debug_mode:
        logger = TensorBoardLogger(
            save_dir=logs_dir,
            name=args.exp_name,
            version=0
        )
    else:
        # log to wandb before training
        logger = WandbLogger(project="live-portrait-train",
                                   log_model="None",
                                   name=args.exp_name+ "_" + exp_prefix_name)


    # Initialize model
    model = LitAutoEncoder(args=args)

    train_dataset = CustomDataset(val_mode=False, debug_mode=args.debug_mode, cache_dir=args.cache_dir, landmark_selected_index=args.landmark_selected_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_dataset = CustomDataset(val_mode=True, debug_mode=args.debug_mode, cache_dir=args.cache_dir, landmark_selected_index=args.landmark_selected_index)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=True)

    # Lightning will automatically use all available GPUs!
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        every_n_epochs=args.every_n_epochs,
        save_top_k=-1,  # Save all checkpoints
        filename='{epoch}-{train_loss:.2f}'  # Customize filename pattern
    )

    trainer = L.Trainer(
        strategy = "ddp_find_unused_parameters_true",
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint_path if args.pretrained_mode == 1 else None)
