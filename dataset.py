import os
import cv2
from torch.utils.data import Dataset


class MultiVideoDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.video_filepaths = self.get_all_video_addresses()
        self.transforms = transforms
        self.video_metadata = []
        self.cumulative_frames = []

        # Store video metadata (frame count, etc.)
        total_frames = 0
        for video_path in self.video_filepaths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_metadata.append({
                'path': video_path,
                'frame_count': frame_count
            })
            total_frames += frame_count
            self.cumulative_frames.append(total_frames)
            cap.release()

        self.total_frames = total_frames

    def get_all_video_addresses(self):
        video_dirs = os.listdir(self.root_dir)
        video_filepaths = []
        for video_dir in video_dirs:
            video_filepaths += self.extract_all_subdirs(video_dir)
        return video_filepaths

    def extract_all_subdirs(self, video_dir):
        root_subdir = os.path.join(self.root_dir, video_dir, "videos")
        subdirs = os.listdir(root_subdir)
        video_filepaths = []

        for subdir in subdirs:
            full_subdir = os.path.join(root_subdir, subdir)
            video_filenames = [filename for filename in os.listdir(full_subdir)]
            video_filenames = [filename for filename in video_filenames]
            video_filepaths += [os.path.join(root_subdir, video_filename)
                                for video_filename in video_filenames]
        return video_filepaths

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        video_idx, frame_idx = self._get_video_index(idx)
        cap = cv2.VideoCapture(self.video_metadata[video_idx]['path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame at index {idx} in video "
                             f"{self.video_metadata[video_idx]['path']}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        return frame

    def __len__(self):
        return self.total_frames

    def _get_video_index(self, idx):
        for video_idx, cumulative_frame in enumerate(self.cumulative_frames):
            if idx < cumulative_frame:
                cum_start_frame_idx = cumulative_frame \
                        - self.video_metadata[video_idx]["frame_count"]
                cum_frame_idx = idx - cum_start_frame_idx
                return cum_frame_idx
        raise IndexError("Index out of range")
