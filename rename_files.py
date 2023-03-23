import os



path = r"C:\Users\xyz\PycharmProjects\Dice Reader\dataset"
files = os.listdir(path)

counter = 0
for file in files:
    old_name = path + "\\" + file
    new_name = path + "\\dice" + str(counter) + ".jpg"
    os.rename(old_name, new_name)
    counter += 1