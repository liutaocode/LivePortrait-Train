import torch
import cv2
from src.modules.spade_generator import SPADEDecoder
from src.modules.warping_network import WarpingNetwork
from src.modules.motion_extractor import MotionExtractor
from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
from train import LitAutoEncoder
import numpy as np
from src.datasets import prepare_source
import os
import argparse


def process_kp(kp_info):
    bs = kp_info['kp'].shape[0]
    kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]
    kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]
    kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]
    kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)
    kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)

    kp = kp_info['kp']
    scale = kp_info['scale'].unsqueeze(-1)
    R = get_rotation_matrix(kp_info['pitch'], kp_info['yaw'], kp_info['roll'])
    exp = kp_info['exp']
    t = kp_info['t'].unsqueeze(1)

    return kp, scale, R, exp, t

def inference(args, checkpoint_path, source_img_path, target_img_path, pretrained_from_liveportrait=False):

    # Initialize Lightning model
    model = LitAutoEncoder(args=args, pretrained_from_liveportrait=pretrained_from_liveportrait)

    if not pretrained_from_liveportrait:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    appearance_feature_extractor = model.appearance_feature_extractor
    motion_extractor = model.motion_extractor
    warping_module = model.warping_module
    spade_generator = model.spade_generator

    # Set models to eval mode
    appearance_feature_extractor.eval()
    motion_extractor.eval()
    warping_module.eval()
    spade_generator.eval()

    source_img = cv2.imread(source_img_path)
    target_img = cv2.imread(target_img_path)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    source_img_256 = cv2.resize(source_img, (256, 256))
    target_img_256 = cv2.resize(target_img, (256, 256))

    source_img = prepare_source(source_img_256)[None]
    target_img = prepare_source(target_img_256)[None]

    with torch.no_grad():
        # Extract features from source image
        f_s = appearance_feature_extractor(source_img)
        x_s_info = motion_extractor(source_img)
        x_s_kp, x_s_scale, x_s_R, x_s_exp, x_s_t = process_kp(x_s_info)
        x_s_full = x_s_scale * (x_s_kp @ x_s_R + x_s_exp) + x_s_t

        # Extract features from target image
        x_t_info = motion_extractor(target_img)
        x_t_kp, x_t_scale, x_t_R, x_t_exp, x_t_t = process_kp(x_t_info)
        x_d_full = x_t_scale * (x_t_kp @ x_t_R + x_t_exp) + x_t_t

        # Warp features
        ret_dct = warping_module(f_s, kp_source=x_s_full, kp_driving=x_d_full)

        # Generate final image
        output_result = spade_generator(feature=ret_dct['out'])

    # print(output_result)
    # Convert output tensor to image
    output_image = np.transpose(output_result.data.cpu().numpy(), [0, 2, 3, 1])[0]  # 1x3xHxW -> HxWx3
    output_image = np.clip(output_image, 0, 1)  # clip to [0,1]
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)  # [0,1] -> [0,255]

    return output_image




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--gan_g_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)
    parser.add_argument("--exp_name", type=str, default="live-portrait-gan")
    parser.add_argument("--saved_to", type=str, default='./outputs/predictions/')
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/xxx.ckpt")
    parser.add_argument("--source_img_path", type=str, default="./test_imgs/source.jpg")
    parser.add_argument("--target_img_path", type=str, default="./test_imgs/target.jpg")


    args = parser.parse_args()

    os.makedirs(args.saved_to, exist_ok=True)

    official_output_image = inference(args, '', args.source_img_path, args.target_img_path, pretrained_from_liveportrait=True)
    pred_output_image = inference(args, args.checkpoint_path, args.source_img_path, args.target_img_path, pretrained_from_liveportrait=False)

    source_img_orig = cv2.imread(args.source_img_path)
    target_img_orig = cv2.imread(args.target_img_path)
    source_img_orig = cv2.cvtColor(source_img_orig, cv2.COLOR_BGR2RGB)
    target_img_orig = cv2.cvtColor(target_img_orig, cv2.COLOR_BGR2RGB)

    # Resize to 512x512
    source_img_512 = cv2.resize(source_img_orig, (512, 512))
    target_img_512 = cv2.resize(target_img_orig, (512, 512))

    # Concatenate horizontally
    concat_img = np.concatenate([source_img_512, target_img_512, official_output_image, pred_output_image], axis=1)

    # Save concatenated result
    cv2.imwrite(os.path.join(args.saved_to, f'{i}.jpg'), concat_img[..., ::-1])  # Convert RGB to BGR for cv2
