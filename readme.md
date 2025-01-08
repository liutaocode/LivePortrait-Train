# Training Code For LivePortrait

Currently this is a basic version, not fully implemented, may contain bugs, please use with caution

TODO:
- Wing loss
- Cascade loss
- Stage 2 & 3 implementation

Completed:
- Stage 1 Training
- PyTorch Lightning integration
- Basic GAN implementation 
- Basic Perception Loss implementation (VGG)


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