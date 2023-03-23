import os



path = r"C:\Users\Artkor1\PycharmProjects\Dice Reader\d6d8d10d12_images_dataset_temp\test"
files = os.listdir(path)

counter = 3187
for file in files:
    old_name = path + "\\" + file
    new_name = path + "\\dice" + str(counter) + ".jpg"
    os.rename(old_name, new_name)
    counter += 1