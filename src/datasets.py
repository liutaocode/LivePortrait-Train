import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import pickle

def prepare_source(img: np.ndarray) -> torch.Tensor:
    """ construct the input as standard
    img: HxWx3, uint8, 256x256
    """
    h, w = img.shape[:2]
    x = img.copy()

    if x.ndim == 3:
        x = x.astype(np.float32) / 255.  # HxWx3, normalized to 0~1
    elif x.ndim == 4:
        x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
    else:
        raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
    x = np.clip(x, 0, 1)  # clip to 0~1
    x = torch.from_numpy(x).permute(2, 0, 1)  # HxWx3 -> 3xHxW
    return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, val_mode=False, debug_mode=False, cache_dir=None, db_path_prefix=None):
        os.makedirs(cache_dir, exist_ok=True)

        self.val_mode = val_mode
        self.debug_mode = debug_mode

        frames_dir = os.path.join(db_path_prefix, 'raw_cropped_frames_jpgs')
        landmark_dir = os.path.join(db_path_prefix, 'landmarks')
        face_orientation_dir = os.path.join(db_path_prefix, 'face_orientation')

        self.meta_lists = []
        for db_name in os.listdir(frames_dir):

            if val_mode:
                if db_name not in ['test_db_name',]:
                    continue
            else:
                if db_name not in ['VoxCeleb2', 'VFHQ']:
                    continue

            if val_mode:
                current_db_cache_dir = os.path.join(cache_dir, db_name+f"_val.pkl")
            else:
                current_db_cache_dir = os.path.join(cache_dir, db_name+f"_train.pkl")

            if os.path.exists(current_db_cache_dir):
                self.meta_lists.extend(pickle.load(open(current_db_cache_dir, 'rb')))
                print(f"Loaded {current_db_cache_dir}")
                continue

            else:
                current_db_meta_lists = []
                for clip_name in tqdm(os.listdir(os.path.join(frames_dir, db_name)), desc='Loading clips'):

                    clip_frames_dir = os.path.join(frames_dir, db_name, clip_name)
                    clip_landmarks_dir = os.path.join(landmark_dir, db_name, clip_name+'.txt')
                    clip_face_orientation_dir = os.path.join(face_orientation_dir, db_name, clip_name+'.npy')


                    frame_len = len(os.listdir(clip_frames_dir))
                    landmark_len = len(open(clip_landmarks_dir, 'r').readlines())

                    yaw_pitch_roll = np.load(clip_face_orientation_dir).astype(np.float32)
                    yaw_pitch_roll = np.clip(yaw_pitch_roll, -90, 90)
                    ypr_len = yaw_pitch_roll.shape[0]

                    min_len = min(frame_len, landmark_len, ypr_len)
                    lmd_obj = self.read_landmark_info(clip_landmarks_dir)

                    current_db_meta_lists.append({
                        'db_name': db_name,
                        'clip_frames_dir': clip_frames_dir,
                        'clip_name': clip_name,
                        'lmd_obj': lmd_obj,
                        'yaw_pitch_roll': yaw_pitch_roll,
                        'min_len': min_len,
                    })

                self.meta_lists.extend(current_db_meta_lists)
                pickle.dump(current_db_meta_lists, open(current_db_cache_dir, 'wb'))
                print(f"Saved {current_db_cache_dir}")

        print(f'Total count: {len(self.meta_lists)}')


    def read_landmark_info(self, lmd_path):
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for i, line in enumerate(lmd_lines):
            # Split the coordinates and filter out any empty strings
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:] # do not include the file name in the first row
            lmd_obj = []

            for coord_pair in coords:
                x, y = coord_pair.split('_')
                lmd_obj.append((int(x)/512, int(y)/512))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)



    def __len__(self):
        return len(self.meta_lists)

    def __getitem__(self, idx):

        clip_frames_dir = self.meta_lists[idx]['clip_frames_dir']
        lmd_obj = self.meta_lists[idx]['lmd_obj']
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


        return {
            'source_img': source_img,
            'target_img': target_img,
            'target_img_512': target_img_512,
            'target_ypr': yaw_pitch_roll[target_idx],
            'target_lmd': lmd_obj[target_idx],
        }
