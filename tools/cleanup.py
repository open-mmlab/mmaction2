import glob 
import os 


if __name__ == '__main__':
    run_format = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/**/**/epoch*.pth"
    ckpt_paths = glob.glob(run_format) 
    for path in ckpt_paths:
        print(path)
        os.system(f"rm {path}")