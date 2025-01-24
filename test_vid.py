import torch
import cv2
from train import LitAutoEncoder
import numpy as np
from src.datasets import prepare_source
import os
import argparse
from src.losses import process_kp, process_kp_original
from tqdm import tqdm  # For progress bar
# Make sure MoviePy is installed: pip install moviepy
from moviepy.editor import VideoFileClip

def load_model(args, checkpoint_path=None, livepotrait_mode=False):
    """
    Load a model given the arguments and (optionally) a checkpoint path.
    Args:
        args: Parsed command-line arguments.
        checkpoint_path: Path to the trained checkpoint (can be None if using live portrait mode).
        livepotrait_mode: Boolean flag to switch the pretrained mode for live portrait.
    Returns:
        model: A LitAutoEncoder model loaded onto the correct device and ready for inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Decide pretrained mode
    if livepotrait_mode:
        args.pretrained_mode = 3
    else:
        args.pretrained_mode = 0

    # Create the model
    model = LitAutoEncoder(args=args)
    model.to(device)

    # Load checkpoint only if provided (i.e., for your trained model)
    if (not livepotrait_mode) and (checkpoint_path is not None):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    # Set to eval
    model.appearance_feature_extractor.eval()
    model.motion_extractor.eval()
    model.warping_module.eval()
    model.spade_generator.eval()

    return model

def inference(model, source_img, target_img, args):
    """
    Perform inference given a loaded model, source image tensor, and target image tensor.
    Args:
        model: A pre-loaded LitAutoEncoder model.
        source_img: A tensor (1 x 3 x H x W) for the source image.
        target_img: A tensor (1 x 3 x H x W) for the target image.
        args: Command-line arguments (used for num_bins, etc.).
    Returns:
        output_image: The output image (H x W x 3) in uint8 format.
    """
    appearance_feature_extractor = model.appearance_feature_extractor
    motion_extractor = model.motion_extractor
    warping_module = model.warping_module
    spade_generator = model.spade_generator

    with torch.no_grad():
        # Extract source features
        f_s = appearance_feature_extractor(source_img)
        x_s_info = motion_extractor(source_img)

        if args.num_bins != 1:
            x_s_kp, x_s_scale, x_s_R, x_s_exp, x_s_t = process_kp_original(x_s_info)
        else:
            x_s_kp, x_s_scale, x_s_R, x_s_exp, x_s_t = process_kp(x_s_info)
        x_s_full = x_s_scale * (x_s_kp @ x_s_R + x_s_exp) + x_s_t

        # Extract target features
        x_t_info = motion_extractor(target_img)
        if args.num_bins != 1:
            x_t_kp, x_t_scale, x_t_R, x_t_exp, x_t_t = process_kp_original(x_t_info)
        else:
            x_t_kp, x_t_scale, x_t_R, x_t_exp, x_t_t = process_kp(x_t_info)
        x_d_full = x_t_scale * (x_t_kp @ x_t_R) + x_t_t

        # Warp the source features according to the target keypoints
        ret_dct = warping_module(f_s, kp_source=x_s_full, kp_driving=x_d_full)

        # Generate the final output image
        output_result = spade_generator(feature=ret_dct['out'])

    # Convert output tensor to a uint8 RGB image
    output_image = np.transpose(output_result.data.cpu().numpy(), [0, 2, 3, 1])[0]
    output_image = np.clip(output_image, 0, 1)
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)

    return output_image

def add_audio_to_video(input_video_path, audio_source_path, output_video_path, fps=25):
    """
    Merge the original audio from the driving (target) video with the newly generated (silent) video.
    Args:
        input_video_path: Path to the silent video we just generated (no audio).
        audio_source_path: Path to the original video that has the audio.
        output_video_path: Path for saving the final merged video.
        fps: Frames per second for the output video.
    Note:
        - This approach uses MoviePy. Ensure you have installed it (pip install moviepy).
        - If the lengths differ, the resulting video will end when the shorter of the two ends.
    """
    video_clip = VideoFileClip(input_video_path)
    audio_clip = VideoFileClip(audio_source_path).audio
    final_clip = video_clip.set_audio(audio_clip)

    # Write the final video with audio
    # We keep the fps the same as our input video or what we used in the inference step.
    final_clip.write_videofile(output_video_path, fps=fps)

def process_video(args, source_img_path, target_video_path, checkpoint_path, saved_to):
    """
    Process the target video frame by frame, performing inference using two loaded models:
    1) Live portrait model (no checkpoint, livepotrait_mode=True).
    2) User's trained model (checkpoint, livepotrait_mode=False).
    Then merge the output video (no audio) with the original audio track from the target video.
    Args:
        args: Parsed command-line arguments.
        source_img_path: Path to the source image.
        target_video_path: Path to the target video.
        checkpoint_path: Path to the trained model checkpoint.
        saved_to: Directory for saving the generated videos.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(saved_to, exist_ok=True)

    # Prepare the source image (resize to 256x256 for the model)
    source_img_orig = cv2.imread(source_img_path)
    source_img_orig = cv2.cvtColor(source_img_orig, cv2.COLOR_BGR2RGB)
    source_img_256 = cv2.resize(source_img_orig, (256, 256))
    source_img_tensor = prepare_source(source_img_256)[None].to(device)

    # Load the models only once
    model_official = load_model(args, checkpoint_path=None, livepotrait_mode=True)
    model_pred = load_model(args, checkpoint_path=checkpoint_path, livepotrait_mode=False)

    # Initialize video capture
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {target_video_path}")
        return

    # Retrieve total number of frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback if FPS is unknown

    # Optional: Prepare video writer for the combined output (silent)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path_silent = os.path.join(saved_to, 'output_silent.mp4')
    out_silent = None

    
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # Convert frame from BGR to RGB
            target_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for model inference
            target_frame_256 = cv2.resize(target_frame_rgb, (256, 256))
            target_img_tensor = prepare_source(target_frame_256)[None].to(device)

            # Inference with the loaded models
            official_output_image = inference(model_official, source_img_tensor, target_img_tensor, args)
            pred_output_image = inference(model_pred, source_img_tensor, target_img_tensor, args)

            # (Optional) Concatenate all images side by side for visualization
            source_img_512 = cv2.resize(source_img_orig, (512, 512))
            target_img_512 = cv2.resize(target_frame_rgb, (512, 512))
            official_output_512 = cv2.resize(official_output_image, (512, 512))
            pred_output_512 = cv2.resize(pred_output_image, (512, 512))

            concat_img = np.concatenate([source_img_512, 
                                         target_img_512, 
                                         official_output_512, 
                                         pred_output_512], axis=1)
            concat_img_bgr = concat_img[..., ::-1]  # Convert RGB -> BGR for OpenCV

            # Initialize the video writer once we know dimensions
            if out_silent is None:
                height, width, _ = concat_img_bgr.shape
                out_silent = cv2.VideoWriter(
                    out_path_silent,
                    fourcc,
                    fps,
                    (width, height)
                )

            out_silent.write(concat_img_bgr)
            pbar.update(1)

    cap.release()
    if out_silent is not None:
        out_silent.release()

    # Now add audio from the original target video
    final_output_with_audio = os.path.join(saved_to, 'output_with_audio.mp4')
    add_audio_to_video(
        input_video_path=out_path_silent,
        audio_source_path=target_video_path,
        output_video_path=final_output_with_audio,
        fps=fps
    )

    print("Video processing completed.")
    print(f"Silent video saved to: {out_path_silent}")
    print(f"Final video with audio saved to: {final_output_with_audio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_img_path", type=str, default="./assets/examples/driving/d19.jpg")
    parser.add_argument("--target_video_path", type=str, default="./assets/examples/video/video_example.mp4")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/your_trained_model.ckpt")
    parser.add_argument("--saved_to", type=str, default="./outputs/predictions/")
    parser.add_argument("--num_bins", type=int, default=66)
    parser.add_argument("--gan_multi_scale_mode", type=bool, default=False)

    args = parser.parse_args()
    args.inference_mode = True

    process_video(
        args,
        args.source_img_path,
        args.target_video_path,
        args.checkpoint_path,
        args.saved_to
    )
