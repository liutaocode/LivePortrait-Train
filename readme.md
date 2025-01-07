# Training Code For LivePortrait

Currently this is a basic version, not fully implemented, may contain bugs, please use with caution

TODO:
- Wing loss
- Cascade loss
- Stage 2 & 3 implementation

Completed:
- PyTorch Lightning integration
- Basic GAN implementation (VGG)
- Basic Perception Loss implementation

Scripts:
- Train.py: Training script
- Test.py: Inference script that outputs images in the following order:
  1. Source image
  2. Driving image
  3. Output from official LivePortrait model
  4. Output from your trained model

References:
- https://github.com/KwaiVGI/LivePortrait
- https://github.com/zhengkw18/face-vid2vid
- https://github.com/wyhsirius/LIA
