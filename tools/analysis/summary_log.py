

def summary_2d_log():
    log_file = '/home/jaeguk/workspace/mmaction2/work_dirs/'
    log_file += 'faster_rcnn_r50_fpn_2x_coco/'
    log_file += '20220824_182250.log'

    best_mAP = [0, 0]
    best_mAP50 = [0, 0]
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if 'bbox_mAP' in line:
                epoch = line.split('[')[1].split(']')[0]
                mAP = float(line.split('bbox_mAP: ')[1].split(',')[0])
                mAP50 = float(line.split('bbox_mAP_50: ')[1].split(',')[0])
                print(epoch, mAP, mAP50)
                if mAP > best_mAP[1]:
                    best_mAP = [epoch, mAP]
                if mAP50 > best_mAP50[1]:
                    best_mAP50 = [epoch, mAP50]
        print(best_mAP)
        print(best_mAP50)

def summary_3d_log():
    log_file = '/home/jaeguk/workspace/mmaction2/work_dirs/'
    log_file += 'ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/'
    log_file += 'JHMDB-tiny/20220829_163210.log'

    best_mAP = [0, 0]
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if 'mAP@0.5IOU:' in line:
                mAP = line[:-1].split('mAP@0.5IOU:')[1]
                epoch = line.split('[')[1].split(']')[0]
                epoch = int(epoch)
                mAP = float(mAP)
                if mAP > best_mAP[1]:
                    best_mAP = [epoch, mAP]

    print(best_mAP)

if __name__ == '__main__':
    summary_3d_log()
