from PIL import Image
import pickle
import os
import numpy as np

train_data = []
test_data = []
size = 19

for filename in os.listdir("./train/face"):    
    img = Image.open(os.path.join("./train/face", filename))
    img_data = np.array(img.getdata())
    img_data = img_data.reshape(size, size)
    train_data.append((img_data, 1))

for filename in os.listdir("./train/non-face"):  
    img = Image.open(os.path.join("./train/non-face", filename))
    img_data = np.array(img.getdata())
    img_data = img_data.reshape(size, size)
    train_data.append((img_data, 0))

# print(train_data)

for filename in os.listdir("./test/face"):
    img = Image.open(os.path.join("./test/face", filename))
    img_data = np.array(img.getdata())
    img_data = img_data.reshape(size, size)
    test_data.append((img_data, 1))

for filename in os.listdir("./test/non-face"):
    img = Image.open(os.path.join("./test/non-face", filename))
    img_data = np.array(img.getdata())
    img_data = img_data.reshape(size, size)
    test_data.append((img_data, 0))

with open("train.pkl", "wb") as file:
    pickle.dump(train_data, file)

with open("test.pkl", "wb") as file:
    pickle.dump(test_data, file)

# for debug

with open("train.pkl", "rb") as file:
    new_train_data = pickle.load(file)

with open("test.pkl", "rb") as file:
    new_test_data = pickle.load(file)

print(f"train_data has length {len(new_train_data)}")
print(f"test_data has length {len(new_test_data)}") 








