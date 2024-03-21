
import jax
from latentvideodiffusion.frame_extractor import FrameExtractor
import cProfile
import numpy as np
import pstats


def test_frame_extractor(directory_path, batch_size, key_seed):
    key = jax.random.PRNGKey(key_seed)
    with FrameExtractor(directory_path, batch_size, key) as extractor:
        # Iterate over the frame extractor and display the frames
        for batch in extractor:
            for i, frame in enumerate(batch):
                # Convert the frame to a format suitable for displaying with OpenCV
                frame_disp = np.array(frame.transpose(2, 1, 0))
                # cv2.imshow(f'Frame {i}', frame_disp)

            # Wait for a key press and then close the windows
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            break # Remove this line if you want to iterate over multiple batches



def main() -> None:
    directory_path = "/mnt/disks/persist/vidmod/data/training_resize"
    batch_size = 128
    key_seed = 1701 
    test_frame_extractor(directory_path, batch_size, key_seed)  

    


if __name__ == '__main__' : 
    cProfile.run('main()','./profile_results')
    p = pstats.Stats('./profile_results')
    p.sort_stats('cumulative').print_stats(10)