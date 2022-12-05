import numpy as np
from PIL import Image
from keras.models import load_model
from ANN import build_ANN
from ANN import val_2_tensor
from ANN import apply
from ANN import find
from skimage import io
path = 'D:/new_dataset/'

# creating model
# new_model=build_ANN(path, 128, 128, True, True, True)
# new_model.save('gfgModel.h5')

savedModel = load_model('gfgModel.h5')

for i in range(200,600):
    Y=apply(i,savedModel)
    img = Image.fromarray(Y.astype('uint8'))
    img.convert('RGB').save(r"D:\cut_image\image" + str(i) + ".png")

for i in range(200,600):
    img=io.imread(r"D:\cut_image\image"+str(i)+".png")
    print(find(img,savedModel))