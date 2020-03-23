import cv2
import numpy as np
import os

#이미지 불러와서 정규화
x = []
y = []
categories = []
category_num = 0

dataset_dir = './dataset'
category_list = os.listdir(dataset_dir)

for category in category_list:
    categories.append(category)
    image_list = os.listdir(dataset_dir + '/' + category)
    for image in image_list:
        image = cv2.imread(dataset_dir + '/' + category + '/' + image)
        image = image / 255.0
        x.append(image.astype(np.float16))
        y.append(category_num)
    category_num += 1

x = np.array(x)
y = np.array(y)
categories = np.array(category)

#train, test set 분리
data_num = x.shape[0]
ratio = 0.8
s = np.arange(data_num)
np.random.shuffle(s)
x = x[s]
y = y[s]

train_x = x[:int(data_num * ratio)]
train_y = y[:int(data_num * ratio)]
test_x = x[int(data_num * ratio) + 1:]
test_y = y[int(data_num * ratio) + 1:]

#저장
np.save("train_x", train_x)
np.save("train_y", train_y)
np.save("test_x", test_x)
np.save("test_y", test_y)
np.save("categories", categories)
