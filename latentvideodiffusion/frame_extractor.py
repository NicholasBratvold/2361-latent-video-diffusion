import cv2
import jax
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        print("started next (Nick)")
        time_start = time.time()
        self.key, idx_key = jax.random.split(self.key)
        # idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames)
        
        local_idx = 0
        video_idx = 0
        frames = []

        jit_randint = jax.jit(jax.random.randint, static_argnames=['shape'])
        _ = jit_randint(idx_key, shape=(100,), minval=0, maxval=100)

        idx_array = jit_randint(idx_key, (self.batch_size,), 0, self.total_frames)
        
        for global_idx in idx_array:
            if(global_idx < self.video_gbl_idxs[0]):
                local_idx = int(global_idx)
                #frame from video 0
            else:
                video_idx = np.searchsorted(self.video_gbl_idxs, int(global_idx))
                local_idx = int(global_idx) - int(self.video_gbl_idxs[video_idx-1])
            
            print(f"video_idx : {video_idx}, local_idx : {local_idx}")
            vid_pth = self.video_files[video_idx]
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
        print("ended next (Nick)")
        print("time (Nick):", time_end-time_start)

        return array.transpose(0,3,2,1)

class FrameExtractor:
    def __init__(self, directory_path, batch_size, key, target_size=(512,300)):
        self.directory_path = directory_path
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi'))] # Adjust as needed
        self.batch_size = batch_size
        self.key = key
        self.frame_counts = [int(cv2.VideoCapture(os.path.join(directory_path, f)).get(cv2.CAP_PROP_FRAME_COUNT)) for f in self.video_files]
        self.cumsum_frames = np.cumsum(self.frame_counts)
        self.total_frames  = self.cumsum_frames[-1]
        self.cap = None
        self.target_size = target_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self

    @jax.profiler.annotate_function
    def __next__(self):
        print("started next (Base)")
        time_start = time.time()

        self.key, idx_key = jax.random.split(self.key) # split PRNG key into 2 new keys
        
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames) # sample uniform random values in [0, self.total_frames)
        
        # jit_randint = jax.jit(jax.random.randint, static_argnames=['shape'])
        # _ = jit_randint(idx_key, shape=(100,), minval=0, maxval=100)

        # idx_array = jit_randint(idx_key, (self.batch_size,), 0, self.total_frames)
        
        frames = []
        # global = across all videos, local = within a video
        for global_idx in idx_array:
            # find local index
            video_idx = bisect.bisect_right(self.cumsum_frames, global_idx) - 1
            local_idx = int(global_idx) - self.cumsum_frames[video_idx]

            print(f"video_idx : {video_idx}, local_idx : {local_idx}")
                
            self.cap = cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx]))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
            ret, frame = self.cap.read()
            self.cap.release()

            if ret:
                # resize video to specified target size
                # frame = cv2.resize(frame, self.target_size)
                frames.append(frame)

        array = jax.numpy.array(frames)
        time_end = time.time()
        print("ended next (Base)")
        print("time (Base):", time_end-time_start)
        
        return array.transpose(0,3,2,1)




def test_frame_extractor(directory_path, batch_size, key_seed):
    key = jax.random.PRNGKey(key_seed)
    
    
    
    # with FrameExtractor(directory_path, batch_size, key) as extractor:
    #     # Iterate over the frame extractor and display the frames
    #     for batch in extractor:
    #         for i, frame in enumerate(batch):
                
    #             frame.block_until_ready()
                
    #             frame_disp_base = np.array(frame.transpose(2, 1, 0))
                
    #             im = Image.fromarray(frame_disp_base).save("/home/vidmod/lvd-dev/lvd-work-dir/lvd-arjun/tests/base.jpg")
                
    #         break # Remove this line if you want to iterate over multiple batches
   
    with FrameExtractorNick(directory_path, batch_size, key) as extractor:
        # Iterate over the frame extractor and display the frames
        for batch in extractor:
            for i, frame in enumerate(batch):
                
                frame.block_until_ready()
                
                frame_disp_nick = np.array(frame.transpose(2, 1, 0))
                im = Image.fromarray(frame_disp_nick).save("/home/vidmod/lvd-dev/lvd-work-dir/lvd-arjun/tests/nick.jpg")
                
            break # Remove this line if you want to iterate over multiple batches
    
    return

def test_frame_extractor_jit(directory_path, batch_size, key_seed):
    key = jax.random.PRNGKey(key_seed)
    with FrameExtractor(directory_path, batch_size, key,jit=True) as extractor:
        # Iterate over the frame extractor and display the frames
        for batch in extractor:
            for i, frame in enumerate(batch):
                # Convert the frame to a format suitable for displaying with OpenCV
                frame.block_until_ready()
                frame_disp = np.array(frame.transpose(2, 1, 0))
               
                
            break # Remove this line if you want to iterate over multiple batches

def main() -> None:
    directory_path = "/mnt/disks/persist/vidmod/data/training_resize"
    batch_size = 120
    key_seed = 800 
    test_frame_extractor(directory_path, batch_size, key_seed)  

    
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        test_frame_extractor(directory_path, batch_size, key_seed)

    
    # cProfile.run('re.compile("foo|bar")')


   


if __name__ == '__main__' : 
    # cProfile.run('main()','/home/vidmod/lvd-dev/lvd-work-dir/lvd-arjun/tests/results')
    main()