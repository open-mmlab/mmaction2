import os 
import glob 

def get_best_val_from_run_path(json_log):
    with open(json_log, 'r') as f:
        lines = f.readlines() 
    
    best_val =  - float('inf')
    epoch = 0
    for dict_str in lines:
        dict_str = dict_str.rstrip() 
        dict_str = dict_str.replace("null", 'None')
        epoch_dict = eval(dict_str) 
        
        if "mode" in epoch_dict and epoch_dict["mode"] == "val":
            if epoch_dict["top1_acc"] > best_val:
                best_val = epoch_dict["top1_acc"]
                epoch = epoch_dict["epoch"]
    
    return {'best_eval': best_val, 'epoch': epoch} 

def get_average_of_last_three_val(json_log):
    with open(json_log, 'r') as f:
        lines = f.readlines() 
    
    all_vals = [] 
    for dict_str in lines:
        dict_str = dict_str.rstrip() 
        dict_str = dict_str.replace("null", 'None')
        epoch_dict = eval(dict_str) 
        
        if "mode" in epoch_dict and epoch_dict["mode"] == "val":
            all_vals.append(epoch_dict["top1_acc"])
    
    average_last_three = sum(all_vals[-3:]) / 3 
    return average_last_three

if __name__ == '__main__':
    # run_folder_pattern = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/vcop/**/*.log.json" 
    run_folder_pattern = "/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/*-resfrozen-us/**/*.log.json"
    folder_paths = glob.glob(run_folder_pattern, recursive=True) 
    for path in folder_paths:
        run_name = path.split('/')[-2] 
        model_name = path.split('/')[-3]
        best_val = get_best_val_from_run_path(path) 
        print(f"Model Name: {model_name} Run Name: {run_name} - Best Val: {best_val}")
