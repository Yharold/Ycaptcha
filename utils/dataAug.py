import os
import Augmentor


def get_zoom_distortion_pipeline(input_path,output_path, num):
    p = Augmentor.Pipeline(input_path,output_path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=0.5, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p


def get_zoom_distortion_tilt_pipeline(input_path,output_path, num):
    p = Augmentor.Pipeline(input_path,output_path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=0.5, grid_width=6, grid_height=2, magnitude=3)
    p.skew_tilt(probability=0.5, magnitude=0.02)
    p.skew_left_right(probability=0.5, magnitude=0.02)
    p.skew_top_bottom(probability=0.5, magnitude=0.02)
    p.sample(num)
    return p


if __name__ == "__main__":
    path = r"E:\Code\Ycaptcha\datasets\train"
    times = 2
    num = len(os.listdir(path)) * times
    # output = r"E:\Code\Ycaptcha\datasets\auged_train_0"
    # p = get_zoom_distortion_pipeline(path,output, num)
    output = r"E:\Code\Ycaptcha\datasets\auged_train_1"
    p = get_zoom_distortion_tilt_pipeline(path,output, num)
