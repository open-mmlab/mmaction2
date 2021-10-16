import os 
from itertools import product


if __name__ == '__main__':
    # mutlisource runs 
    # train = ['d1d2', 'd2d3', 'd1d3'] 
    # test = ['d3', 'd2'] 

    # run_command_format = "./tools/dist_train.sh /home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_{}_{}_ekmmsada_rgb.py 3 --cfg-options work_dir=./work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/train_{}_test_{}/ --validate"
    # folder = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/train_{}_test_{}/"
    # for tr, te in list(zip(train, test)):
    #     if not os.path.exists(folder.format(tr, te)):
    #         command = run_command_format.format(tr, te, tr, te) 
    #         print(command)
    #         os.system(command)
    #     else:
    #         print('Already Computed')



    # single source runs 
    train = ['D1', 'D2', 'D3'] 
    test = ['D2', 'D3'] 

    run_command_format = "./tools/dist_train.sh /home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_rgb.py 3 --cfg-options work_dir=./work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/train_[{}]_test_[{}]/ data.train.domain='{}' data.val.domain='{}' --validate "
    folder = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/train_[{}]_test_[{}]/"
    for tr, te in list(product(train, test)):
        if not os.path.exists(folder.format(tr, te)):
            command = run_command_format.format(tr, te, tr, te) 
            print(command)
            os.system(command)
        else:
            # 
            print('removing!')
            rm_cmd = f"rm -rf {folder.format(tr, te)}"
            print(rm_cmd)
            os.system(rm_cmd)
