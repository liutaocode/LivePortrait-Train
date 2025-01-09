# Training Code For LivePortrait

> This is a project for training LivePortrait, currently this is a basic version, not fully implemented, may contain bugs, please use with caution

### TODO:
- Cascade loss
- Stage 2 & 3 implementation

### Completed:
- Stage 1 Training
- PyTorch Lightning integration
- Basic GAN implementation 
- Basic Perception Loss implementation (VGG)
- Wing loss (under testing)

With completed part, you can train a basic version of stage 1 of LivePortrait.

### Wing loss

Since LivePortrait did not disclose the specific 10 keypoints information, I currently use the following 10 keypoint indices:
Eye Region (6 points):

Left Eye: 36(outer corner), 39(inner corner), 37(upper eyelid midpoint)
Right Eye: 42(outer corner), 45(inner corner), 43(upper eyelid midpoint)

Mouth Region (4 points):

48(left corner)
54(right corner) 
51(upper lip midpoint)
57(lower lip midpoint)


Scripts:
- Train.py: Training script for stage 1
- Test.py: Inference script that outputs images in the following order:
  1. Source image
  2. Driving image
  3. Output from official LivePortrait model
  4. Output from your trained model


Due to certain constraints, I currently have no plan to release pre-trained models.

References:
- https://github.com/KwaiVGI/LivePortrait
- https://github.com/zhengkw18/face-vid2vid
- https://github.com/wyhsirius/LIA