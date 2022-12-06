
from PIL import Image
from keras.models import load_model
from ANN import build_ANN
from ANN import val_2_tensor
from ANN import apply
from ANN import find
from skimage import io

path = r"C:\Users\user\PycharmProjects\image_hamming\DATA"


def CreateModel(pathToImages, nameToSaveModel, size):
    model = build_ANN(pathToImages, size, size, use_flip=False, use_rotate=False, extend_flag=False)
    model.save(nameToSaveModel)


def UseModel(modelName, fromRange, toRange, pathToSave):
    savedModel = load_model(modelName)
    for i in range(fromRange, toRange):
        Y = apply(i, savedModel)
        img = Image.fromarray(Y.astype('uint8'))
        img.convert('RGB').save(pathToSave + "\img_" + str(i) + ".png")


def ReverseModel(modelName, fromRange, toRange, imagesPath):
    savedModel = load_model(modelName)
    for i in range(fromRange, toRange):
        img = io.imread(imagesPath + "\img_" + str(i) + ".png")
        ind = find(img, savedModel)
        print("i = " + str(i) + ";   IND = " + str(ind))


if __name__ == "__main__":
    modelName = path + "\gfgModel.h5"

    # ------ создание
    CreateModel(path + "/IN/", modelName, 512)

    # ----- использование
    UseModel(modelName, 1, 35, path + "\OUT")

    # ------ реверс
    # ReverseModel(modelName, 11, 15, path + "\OUT" )
