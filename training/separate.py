import os
import shutil
import random

SOURCE_PATH = '../datasets/preprocessout'
LABEL_PATH = '../datasets/labels'
TRAIN_PATH = '../datasets/train'
VALID_PATH = '../datasets/valid'

percentage = 0.3

names = os.listdir(SOURCE_PATH)
for name in names:
    img_path = os.path.join(SOURCE_PATH, name)
    slabel_path = os.path.join(LABEL_PATH, name)
    img_names = os.listdir(img_path)

    valid_size = round(len(img_names) * percentage)
    valid_set = random.sample(range(len(img_names)), valid_size)
    print("folder: " + name + ", size=" + str(len(img_names)) + ", valid=" + str(valid_size))

    index_v = 0

    for img_name in img_names:
        if int(img_name.split(".jpg")[0]) == valid_set[index_v]:
            save_path = os.path.join(VALID_PATH, name)
            index_v += 1
        else:
            save_path = os.path.join(TRAIN_PATH, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy2(img_path + '/' + img_name, save_path + '/' + img_name)
        shutil.copy2(slabel_path + '/' + img_name.split(".jpg")[0]+'.txt', save_path + '/' + img_name.split(".jpg")[0]+'.txt')