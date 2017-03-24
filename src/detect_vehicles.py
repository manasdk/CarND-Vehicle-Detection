import os
import random

import cv2

from window import SubSampleWindowSearch
from moviepy.editor import VideoFileClip

MODULE_DIR = os.path.dirname(__file__)


class DetectVehicles:
    """
    Vehicle detection pipeline entry point. This implementation is less pipeline comprising of
    stages and more of a composition approach for sake of simplicity.

    The job of this stage is to read a video frame-by-frame and processes each frame individually
    and write the frame out to an output video.
    """

    def __init__(self, input_video_path, output_video_path):
        self.per_frame_car_detector = SubSampleWindowSearch()
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.frame_write_percentage = 10
        self.write_idx = 0

    def run(self):
        """
        Read input video and spit out an output video to the filesystem.
        """
        input_clip = VideoFileClip(self.input_video_path)
        # input_clip.set_mask(0)
        print('Using multi-scale windows search to detect cars')
        output_clip = input_clip.fl_image(self._process_frame_2)
        output_clip.write_videofile(self.output_video_path, audio=False)

    def _process_frame(self, img):
        """
        Uses hog sub-sampling window search to detect cars in a frame.
        """
        return self.per_frame_car_detector.find_cars(
            img=img, ystart=400, ystop=656, orient=9, pix_per_cell=8, cell_per_block=2,
            spatial_size=(32, 32), hist_bins=32
        )

    def _process_frame_2(self, img):
        """
        Uses multi-scale window search to detect cars in a frame
        """
        # image here is in BGR space and that causes the classifer to fail since it is trained
        # against YUV space images.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_img = self.per_frame_car_detector.find_cars_in_windows_scaled(
            img=img, cspace='YUV', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
            cell_per_block=2
        )
        self._try_write_image(img, out_img)
        return out_img

    def _try_write_image(self, in_frame, out_frame):
        write_decision = random.randint(0, 100)
        if write_decision >= self.frame_write_percentage:
            return
        out_path_input = os.path.join(MODULE_DIR, '../video_output_images/f%d_ip.jpg' % self.write_idx)
        out_path_output = os.path.join(MODULE_DIR, '../video_output_images/f%d_op.jpg' % self.write_idx)
        cv2.imwrite(out_path_input, in_frame)
        cv2.imwrite(out_path_output, out_frame)
        self.write_idx += 1


def process_video(input_video_file, output_video_file):
    input_video_path = os.path.join(MODULE_DIR, input_video_file)
    output_video_path = os.path.join(MODULE_DIR, output_video_file)
    vehicle_detector = DetectVehicles(input_video_path, output_video_path)
    vehicle_detector.run()


if __name__ == '__main__':
    # project video
    process_video(
        input_video_file='../project_video.mp4', output_video_file='../project_video_output_3.mp4'
    )
    # test video
    # process_video(
    #     input_video_file='../test_video.mp4', output_video_file='../test_video_output.mp4'
    # )
