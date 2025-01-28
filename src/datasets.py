import torch
import numpy as np
import cv2
import os
from tqdm import tqdm


def prepare_source(img: np.ndarray) -> torch.Tensor:
    """
    Convert input uint8 image in [0..255] to a torch.Tensor in [0..1], shape [C,H,W].
    """
    x = img.astype(np.float32) / 255.0  # normalize to [0,1]
    x = np.clip(x, 0, 1)
    x = torch.from_numpy(x).permute(2, 0, 1)  # HxWx3 -> 3xHxW
    return x


def create_eye_mouth_mask(
    landmarks_68: np.ndarray,
    image_size: int = 512,
    # For eyes (optional erosion + dilation)
    eye_erosion_iters: int = 1,
    eye_dilate_iters: int = 1,
    # For mouth
    mouth_dilate_iters: int = 2
    ) -> (np.ndarray, np.ndarray):
    """
    Create binary masks for eyes and mouth based on 68 facial landmarks.
    - Eyes: We use the typical 68-landmark indices for left/right eyes,
            fill them with fillConvexPoly. Then optional erosion/dilation.
    - Mouth: We ONLY use outer lip indices [48..59], ignore the inner ring [60..67].
             Then we compute a convex hull to ensure a smooth boundary (no inward spikes),
             fill that hull, and do morphological dilation to expand the region.

    Args:
        landmarks_68 (np.ndarray): shape (68,2), each row is (x_norm, y_norm) in [0,1].
        image_size (int): final mask size (width=height=image_size).
        eye_erosion_iters (int): how many times to erode eye region before dilate.
        eye_dilate_iters (int): how many times to dilate eye region.
        mouth_dilate_iters (int): how many times to dilate mouth region.
                                  (We skip erosion for mouth in this example.)
    Returns:
        eye_mask  (np.ndarray): shape (image_size, image_size, 1), float32 in [0,1].
        mouth_mask(np.ndarray): shape (image_size, image_size, 1), float32 in [0,1].
    """
    # Initialize empty masks
    eye_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mouth_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Indices for left/right eye in 68-landmarks
    left_eye_idx = [36, 37, 38, 39, 40, 41]
    right_eye_idx = [42, 43, 44, 45, 46, 47]

    # Outer lips only: [48..59], ignoring [60..67] (inner lips)
    outer_mouth_idx = list(range(48, 60))

    # Convert normalized coords -> pixel coords
    def to_px_coords(idx_list):
        return [
            (int(landmarks_68[i, 0] * image_size),
             int(landmarks_68[i, 1] * image_size))
            for i in idx_list
        ]

    left_eye_pts = to_px_coords(left_eye_idx)
    right_eye_pts = to_px_coords(right_eye_idx)
    mouth_pts = to_px_coords(outer_mouth_idx)

    def fill_polygon(mask, pts):
        pts_array = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts_array, 255)

    # Fill left eye / right eye
    fill_polygon(eye_mask, left_eye_pts)
    fill_polygon(eye_mask, right_eye_pts)

    # Mouth: use convex hull on outer-lip points => no inward spikes
    mouth_pts_array = np.array(mouth_pts, dtype=np.int32)
    mouth_hull = cv2.convexHull(mouth_pts_array)
    cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)

    # Morphological ops
    kernel= np.ones((7, 7), dtype=np.uint8)

    # Eye region: optional erosion then dilation
    if eye_erosion_iters > 0:
        eye_mask = cv2.erode(eye_mask, kernel, iterations=eye_erosion_iters)
    if eye_dilate_iters > 0:
        eye_mask = cv2.dilate(eye_mask, kernel, iterations=eye_dilate_iters)

    # Mouth region: skip erosion, do dilation to expand outward
    # (use a slightly bigger kernel / iteration for 512 resolution)
    if mouth_dilate_iters > 0:
        mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=mouth_dilate_iters)

    # Convert to float32 binary in [0,1], shape (H,W,1)
    eye_mask = (eye_mask > 0).astype(np.float32)[..., None]
    mouth_mask = (mouth_mask > 0).astype(np.float32)[..., None]

    return eye_mask, mouth_mask

# dataset is not fully implemented here
# TODO: implement the dataset by following your own format
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, val_mode=False, db_path_prefix=None, landmark_selected_index=None):

        assert db_path_prefix is not None, "db_path_prefix is required"
        assert landmark_selected_index is not None, "landmark_selected_index is required"
        assert os.path.exists(db_path_prefix), "db_path_prefix does not exist. We suggest you can follow https://github.com/liutaocode/talking_face_preprocessing"

        self.val_mode = val_mode
        self.landmark_selected_index = landmark_selected_index
        
        self.meta_lists = []
        for db_name in os.listdir(db_path_prefix):

            if not os.path.isdir(os.path.join(db_path_prefix, db_name)):
                continue

            frames_dir = os.path.join(db_path_prefix, db_name, 'raw_cropped_frames')
            print('frames_dir', frames_dir)
            assert os.path.exists(frames_dir), "frames_dir does not exist. please check the db_path_prefix."
            landmark_dir = os.path.join(db_path_prefix, db_name, 'landmarks')
            assert os.path.exists(landmark_dir), "landmark_dir does not exist. please check the db_path_prefix."
            face_orientation_dir = os.path.join(db_path_prefix, db_name, 'face_orientation')
            assert os.path.exists(face_orientation_dir), "face_orientation_dir does not exist. please check the db_path_prefix."


            if val_mode:
                if db_name not in ['db_name_xxxx1',]:
                    continue
            else:
                if db_name not in ['db_name_xxxx1',]:
                    continue

      
            current_db_meta_lists = []
            for clip_name in tqdm(os.listdir(frames_dir), desc='Loading clips'):

                clip_frames_dir = os.path.join(frames_dir, clip_name)
                clip_landmarks_dir = os.path.join(landmark_dir, clip_name+'.txt')
                clip_face_orientation_dir = os.path.join(face_orientation_dir, clip_name+'.npy')


                frame_len = len(os.listdir(clip_frames_dir))
                landmark_len = len(open(clip_landmarks_dir, 'r').readlines())

                yaw_pitch_roll = np.load(clip_face_orientation_dir).astype(np.float32)
                yaw_pitch_roll = np.clip(yaw_pitch_roll, -90, 90)
                ypr_len = yaw_pitch_roll.shape[0]

                min_len = min(frame_len, landmark_len, ypr_len)
                selected_lmd_obj = self.read_landmark_info(clip_landmarks_dir, landmark_selected_index=self.landmark_selected_index)
                all_lmd_obj = self.read_landmark_info(clip_landmarks_dir, landmark_selected_index=None)

                current_db_meta_lists.append({
                    'db_name': db_name,
                    'clip_frames_dir': clip_frames_dir,
                    'clip_name': clip_name,
                    'selected_lmd_obj': selected_lmd_obj,
                    'all_lmd_obj': all_lmd_obj,
                    'yaw_pitch_roll': yaw_pitch_roll,
                    'min_len': min_len,
                })

            self.meta_lists.extend(current_db_meta_lists)

        assert len(self.meta_lists) > 0, "No clips found in the dataset."

        #TODO you must delete this code after you have implemented the dataset
        if len(self.meta_lists) == 1:
             for _ in range(20):
                 self.meta_lists.extend(self.meta_lists) # simulate the dataset
        print(f'Total count: {len(self.meta_lists)}')

    # TODO： make sure the resolution here is the same as the original dataset
    def read_landmark_info(self, lmd_path, landmark_selected_index=None, pixel_scale=512):
        # print('landmark_selected_index', landmark_selected_index)
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for i, line in enumerate(lmd_lines):
            # Split the coordinates and filter out any empty strings
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:] # do not include the file name in the first row
            lmd_obj = []
            if landmark_selected_index is not None and len(landmark_selected_index) > 0:
                # Ensure that the coordinates are parsed as integers
                for idx in landmark_selected_index:
                    coord_pair = coords[idx]
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale, int(y)/pixel_scale))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale, int(y)/pixel_scale))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)



    def __len__(self):
        return len(self.meta_lists)

    def __getitem__(self, idx):

        clip_frames_dir = self.meta_lists[idx]['clip_frames_dir']
        selected_lmd_obj = self.meta_lists[idx]['selected_lmd_obj']
        all_lmd_obj = self.meta_lists[idx]['all_lmd_obj']
        yaw_pitch_roll = self.meta_lists[idx]['yaw_pitch_roll']
        min_len = self.meta_lists[idx]['min_len']

        # Get all jpg files in clip_frames_dir and sort them
        frame_files = sorted([f for f in os.listdir(clip_frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
        frame_paths = [os.path.join(clip_frames_dir, f) for f in frame_files]

        # Randomly select 2 indices from range(min_len)
        if self.val_mode:
            selected_indices = np.array([0, min_len-1])
        else:
            selected_indices = np.random.choice(min_len, size=2, replace=False)

        source_idx, target_idx = selected_indices

        source_img_path = frame_paths[source_idx]
        target_img_path = frame_paths[target_idx]

        source_img = cv2.imread(source_img_path)
        target_img = cv2.imread(target_img_path)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        source_img_256 = cv2.resize(source_img, (256, 256))
        target_img_256 = cv2.resize(target_img, (256, 256))
        target_img_512 = cv2.resize(target_img, (512, 512))

        source_img = prepare_source(source_img_256)
        target_img = prepare_source(target_img_256)
        target_img_512 = prepare_source(target_img_512)

        target_eye_mask, target_mouth_mask = create_eye_mouth_mask(
            all_lmd_obj[target_idx], image_size=512,
            eye_erosion_iters=1, eye_dilate_iters=5,
            mouth_dilate_iters=5
        )

        return {
            'source_img': source_img,
            'target_img': target_img,
            'target_img_512': target_img_512,
            'target_ypr': yaw_pitch_roll[target_idx],
            'source_ypr': yaw_pitch_roll[source_idx],
            'target_lmd': selected_lmd_obj[target_idx],
            'source_lmd': selected_lmd_obj[source_idx],
            'target_eye_mask': target_eye_mask,
            'target_mouth_mask': target_mouth_mask,
        }
