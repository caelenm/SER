import os
import shutil
import random


def split_files(target_directory_sorted, train_directory, test_directory):
    file_list = os.listdir(target_directory_sorted)
    num_files = len(file_list)
    num_train = int(0.8 * num_files)
    train_files = random.sample(file_list, num_train)
    test_files = list(set(file_list) - set(train_files))



    for file in train_files:
        shutil.move(os.path.join(target_directory_sorted, file), train_directory)

    for file in test_files:
        shutil.move(os.path.join(target_directory_sorted, file), test_directory)

    return