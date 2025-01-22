import torch
import torch.nn.functional as F
from torch import nn
from src.utils.camera import get_rotation_matrix
from src.utils.camera import headpose_pred_to_degree

# modified code (change the headpose to mse mode)
def process_kp(kp_info):
    bs = kp_info['kp'].shape[0]

    kp_info['pitch'] = 90.0 * torch.tanh(kp_info['pitch'])
    kp_info['yaw'] = 90.0 * torch.tanh(kp_info['yaw'])
    kp_info['roll'] = 90.0 * torch.tanh(kp_info['roll'])

    # kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
    # kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
    # kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
    kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
    kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

    kp = kp_info['kp']
    scale = kp_info['scale'].unsqueeze(-1)
    R = get_rotation_matrix(kp_info['pitch'], kp_info['yaw'], kp_info['roll'])
    exp = kp_info['exp']
    t = kp_info['t'].unsqueeze(1)

    return kp, scale, R, exp, t

# liveportrait original code (classification mode)
def process_kp_original(kp_info):
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


def multi_scale_g_nonsaturating_loss(fake_pred):
    # fake_pred is a list of discriminator outputs at each scale
    loss = 0
    for scale in fake_pred:  # Iterate through discriminator outputs at each scale
        # Take the last layer output of each scale
        last_layer = scale[-1]
        loss += F.softplus(-last_layer).mean()
    return loss / len(fake_pred)  # Average over all scales

def multi_scale_d_nonsaturating_loss(fake_pred, real_pred):
    loss = 0
    # Iterate through discriminator outputs at each scale
    for scale_fake, scale_real in zip(fake_pred, real_pred):
        # Take the last layer output of each scale
        fake_last = scale_fake[-1]
        real_last = scale_real[-1]

        real_loss = F.softplus(-real_last)
        fake_loss = F.softplus(fake_last)
        loss += real_loss.mean() + fake_loss.mean()

    return loss / len(fake_pred)  # Average over all scales

def single_scale_g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def single_scale_d_nonsaturating_loss(fake_pred, real_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def make_coordinate_grid_2d(shape):
    """Create a meshgrid [-1,1] x [-1,1] of given shape"""
    h, w = shape
    x = torch.arange(w).float()
    y = torch.arange(h).float()

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

class Transform:
    """Random TPS transformation for equivariance constraints."""
    def __init__(self, bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        self.control_points = make_coordinate_grid_2d((points_tps, points_tps))
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean=0, std=sigma_tps * torch.ones([bs, 1, points_tps ** 2]))

    def transform_frame(self, frame):
        """Apply spatial transformation to frame."""
        device = frame.device

        grid = make_coordinate_grid_2d(frame.shape[2:]).unsqueeze(0).to(device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, align_corners=True, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        """Apply transformation to coordinates."""
        theta = self.theta.to(coordinates.device)
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        control_points = self.control_points.to(coordinates.device)
        control_params = self.control_params.to(coordinates.device)
        distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
        distances = torch.abs(distances).sum(-1)

        result = distances ** 2
        result = result * torch.log(distances + 1e-6)
        result = result * control_params
        result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
        transformed = transformed + result

        return transformed

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake_features, real_features):
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j], real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss


class EquivarianceLoss(nn.Module):
    """Enhanced Equivariance loss for keypoint detection"""
    def __init__(self, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        super().__init__()
        self.sigma_affine = sigma_affine
        self.sigma_tps = sigma_tps
        self.points_tps = points_tps

    def forward(self, x_t_256, x_t_full, x_s_kp, motion_extractor):
        """
        Args:
            x_s_256: Source image [B, C, H, W]
            motion_extractor: Motion extraction network
        """
        batch_size = x_t_256.shape[0]

        # 1. Create a random transformation
        transform = Transform(batch_size,
                           self.sigma_affine,
                           self.sigma_tps,
                           self.points_tps)

        # 2. Extract keypoints from original image
        # x_s_info = motion_extractor(x_s_256)
        original_kp = x_t_full.reshape(batch_size, -1, 3)  # BxNx3

        # 3. Apply transformation to image
        transformed_image = transform.transform_frame(x_t_256)

        # uncomment the following code if you want to save transformed image
        # # Save transformed image as PNG
        # # Convert from tensor [B,C,H,W] to numpy [H,W,C]
        # img_np = transformed_image[0].detach().cpu().numpy()  # Take first image from batch
        # img_np = np.transpose(img_np, (1,2,0))  # CHW -> HWC
        # img_np = (img_np * 255).astype(np.uint8)  # Scale from [0,1] to [0,255]
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # RGB -> BGR for cv2
        # i = 1
        # while os.path.exists(f'assets/tps/{i}.png'):
        #     i += 1
        # os.makedirs('assets/tps', exist_ok=True)
        # cv2.imwrite(f'assets/tps/{i}.png', img_np)

        # 4. Extract keypoints from transformed image
        transformed_info = motion_extractor(transformed_image)
        # transformed_kp_info = transformed_info['kp'].reshape(batch_size, -1, 3)  # BxNx3
        _, x_transformed_scale, x_transformed_R, x_transformed_exp, x_transformed_t = process_kp(transformed_info)

        x_transformed_full = x_transformed_scale * (x_s_kp @ x_transformed_R + x_transformed_exp) + x_transformed_t

        # 5. Apply inverse transformation to transformed keypoints
        reverse_transformed_full = transform.warp_coordinates(x_transformed_full[..., :2])  # Only transform x,y coordinates

        # 6. Calculate loss between original and reverse-transformed keypoints
        loss = torch.mean((original_kp[..., :2] - reverse_transformed_full) ** 2)

        return loss

class KeypointPriorLoss(nn.Module):
    def __init__(self, Dt=0.1, zt=0.33):
        super().__init__()
        self.Dt, self.zt = Dt, zt

    def forward(self, kp_d):
        # use distance matrix to avoid loop
        dist_mat = torch.cdist(kp_d, kp_d).square()
        loss = (
            torch.max(0 * dist_mat, self.Dt - dist_mat).sum((1, 2)).mean()
            + torch.abs(kp_d[:, :, 2].mean(1) - self.zt).mean()
            - kp_d.shape[1] * self.Dt
        )
        return loss


class HeadPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()  # nn.L1Loss()

    def forward(self, yaw, pitch, roll, real_yaw, real_pitch, real_roll):
        # Normalize inputs by dividing by 90 degrees
        yaw = yaw / 90.0
        pitch = pitch / 90.0
        roll = roll / 90.0
        real_yaw = real_yaw / 90.0
        real_pitch = real_pitch / 90.0
        real_roll = real_roll / 90.0

        loss = (self.criterion(yaw, real_yaw.detach()) + self.criterion(pitch, real_pitch.detach()) + self.criterion(roll, real_roll.detach())) / 3
        return loss


class DeformationPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_d):
        loss = delta_d.abs().mean()
        return loss

# Original WingLoss
# class WingLoss(nn.Module):
#     def __init__(self, omega=10, epsilon=2):
#         super(WingLoss, self).__init__()
#         self.omega = omega
#         self.epsilon = epsilon

#     def forward(self, pred, target):
#         y = target
#         y_hat = pred
#         delta_y = (y - y_hat).abs()
#         delta_y1 = delta_y[delta_y < self.omega]
#         delta_y2 = delta_y[delta_y >= self.omega]
#         loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
#         C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
#         loss2 = delta_y2 - C
#         return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

# Modified WingLoss
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        delta_y = (pred - target).abs()
        C = self.omega - self.omega * torch.log(torch.tensor(1 + self.omega / self.epsilon))
        loss = torch.where(
            delta_y < self.omega,
            self.omega * torch.log(1 + delta_y / self.epsilon),
            delta_y - C
        )
        return delta_y, loss.mean()
