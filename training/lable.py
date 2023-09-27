import os

SOURCE_PATH = '../datasets/preprocessout'
OUTPUT_PATH = '../datasets/labels'
names = os.listdir(SOURCE_PATH)

count = 0

for name in names:
    path = os.path.join(SOURCE_PATH, name)
    img_names = os.listdir(path)
    for img_name in img_names:
        save_name = img_name.split(".jpg")[0]+'.txt'
        txt_path = os.path.join(OUTPUT_PATH, name)

        # Create the directory if it doesn't exist
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        with open(os.path.join(txt_path, save_name), "w") as f:
            f.write(name)
            print("label created:" + f.name)
            count += 1

print(str(count) + " labels created")