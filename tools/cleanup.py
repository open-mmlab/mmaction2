import glob 
import os 


if __name__ == '__main__':
    run_format = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/**/**/epoch*.pth"
    gif_format = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/**/*.gif"
    # /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/slowfast_contrastive_head/train_D1_test_D2/videos/gradcam_P01_11_3750.gif
    ckpt_paths = glob.glob(run_format) 
    gif_paths = glob.glob(gif_format, recursive=True)
    for path in ckpt_paths:
        print(path)
        os.system(f"rm {path}")
    
    for path in gif_paths:
        print(path)
        os.system(f"rm {path}")
    