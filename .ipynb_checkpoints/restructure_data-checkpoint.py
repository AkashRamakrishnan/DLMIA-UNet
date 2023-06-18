import os
import glob
import numpy as np
import shutil

def restructure_data(source_path, dest_path):
    # Training
    train_dir = os.path.join(source_path, 'training')
    train_image_dest = os.path.join(dest_path, 'train/images')
    train_target_dest = os.path.join(dest_path, 'train/targets')
    ct = 0
    ci = 0
    if not os.path.exists(train_image_dest):
        os.makedirs(train_image_dest)
    if not os.path.exists(train_target_dest):
        os.makedirs(train_target_dest)
    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        if os.path.isfile(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if file_name[-14:-9] == 'frame':
                source_file = os.path.join(folder_path, file_name)
                destination_file = os.path.join(train_image_dest, file_name)
                shutil.copy2(source_file, destination_file)
                ci += 1
            elif file_name[-9:-7] == 'gt':
                source_file = os.path.join(folder_path, file_name)
                destination_file = os.path.join(train_target_dest, file_name[:-10]+file_name[-7:])
                shutil.copy2(source_file, destination_file)
                ct += 1
    print('Transferred {ci} images and {ct} targets in train set'.format(ci=ci, ct=ct))    
    # Testing
    test_dir = os.path.join(source_path, 'testing')
    test_image_dest = os.path.join(dest_path, 'test/images')
    test_target_dest = os.path.join(dest_path, 'test/targets')
    ct = 0
    ci = 0
    if not os.path.exists(test_image_dest):
        os.makedirs(test_image_dest)
    if not os.path.exists(test_target_dest):
        os.makedirs(test_target_dest)
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isfile(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if file_name[-14:-9] == 'frame':
                source_file = os.path.join(folder_path, file_name)
                destination_file = os.path.join(test_image_dest, file_name)
                shutil.copy2(source_file, destination_file)
                ci += 1
            elif file_name[-9:-7] == 'gt':
                source_file = os.path.join(folder_path, file_name)
                destination_file = os.path.join(test_target_dest, file_name[:-10]+file_name[-7:])
                shutil.copy2(source_file, destination_file)
                ct += 1
    print('Transferred {ci} images and {ct} targets in test set'.format(ci=ci, ct=ct))
    
if __name__ == '__main__':
    source_path = "../ACDC/database/"
    dest_path = "data/"
    restructure_data(source_path, dest_path)