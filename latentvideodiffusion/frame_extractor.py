import cv2
import jax
from jax import jit
import timeit
import numpy as np
import os
import cProfile
from PIL import Image
import time
import bisect

class FrameExtractorNick:
    def __init__(self, directory_path, batch_size, key, target_size=(512,300)):
        self.directory_path = directory_path
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi', '.npy'))] # Adjust as needed
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi', '.npy'))] # Adjust as needed
        self.batch_size = batch_size
        self.key = key
        self.video_gbl_idxs = np.zeros(len(self.video_files)) #holds global idx value for every video 
        self.total_frames = 0
        i = 0

        for f in self.video_files:
            if f.endswith('.npy'):
                frame_count = int(np.shape(np.load(os.path.join(directory_path, f)))[0])
            else:
                frame_count = int(cv2.VideoCapture(os.path.join(directory_path, f)).get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames += frame_count
            self.video_gbl_idxs[i] = self.total_frames
            i += 1
        self.cap = None
        self.vid_arr = None
        self.target_size = target_size
        # self.preload_data()
        self.split_jit = jax.jit(jax.random.split)
        self.randomint_jit = jax.jit(jax.random.randint,static_argnames=['shape'])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        

        self.key, idx_key = self.split_jit(self.key)
        idx_array = self.randomint_jit(idx_key, (self.batch_size,), 0, self.total_frames)
        local_idx = 0
        video_idx = 0
        frames = []
        
        for global_idx in idx_array:
            if(global_idx < self.video_gbl_idxs[0]):
                local_idx = int(global_idx)
                #frame from video 0
            else:
                video_idx = np.searchsorted(self.video_gbl_idxs, int(global_idx))
                local_idx = int(global_idx) - int(self.video_gbl_idxs[video_idx-1])
            # print("frame", local_idx)
            # print("video", video_idx)
            vid_pth = self.video_files[video_idx]
            # frame = self.preloaded_data[vid_pth][local_idx]
            # frames.append(frame)
            #Selecting frame for numpy files
            if vid_pth.endswith('.npy'):
                self.vid_arr = np.load(os.path.join(self.directory_path, vid_pth))
                frame = self.vid_arr[local_idx - 1]
                frames.append(frame)
            #Selecting frame for video files
            else:
                self.cap = cv2.VideoCapture(os.path.join(self.directory_path, vid_pth))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                ret, frame = self.cap.read()
                self.cap.release()

                if ret:
                    frames.append(frame)
       
        array = jax.numpy.array(frames)
        time_end = time.time()
        print("ended next (Base)")
        print("time (Base):", time_end- time_start)
        
        return array.transpose(0,3,2,1)
    

def extract_frames(video_path, num_frames, key, target_size=(512, 300)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(str(total_frames)+ " total frames")
    if num_frames > total_frames or num_frames <= 0:
        raise ValueError("Invalid number of frames specified.")

    random_indices = jax.random.randint(key, (num_frames,), 0, total_frames)

    frames = []
    for idx in random_indices:
        ret, frame = cap.read()
        if ret:
            # Resize video to specified target size
            # frame = cv2.resize(frame, target_size)
            frames.append(frame)

    cap.release()

    return jax.numpy.array(frames).transpose(0, 3, 2, 1)