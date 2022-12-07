import os
from PIL import Image
from skimage import io
import ANN

path = r"C:\Users\user\PycharmProjects\image_hamming\DATA"


def CreateModel(pathToImages, nameToSaveModel, sizeVer, sizeHor ):
    model = ANN.BuildANNModel( pathToImages, sizeVer, sizeHor, use_flip=False, use_rotate=False, extend_flag=False)
    model.save(nameToSaveModel)


def UseModel(modelName, fromRange, toRange, pathToSave ):
    ( savedModel, savedImagesNumber, verSize, horSize, layersNum) = ANN.LoadANNModel(modelName) # savedModel = load_model(modelName)
    for i in range(fromRange, toRange + 1):
        Y = ANN.ApplyANNModel( savedModel, i,  verSize, horSize )
        img = Image.fromarray(Y.astype('uint8'))
        img.convert('RGB').save(pathToSave + "\img_" + str(i) + ".png")


def ReverseModel( modelName, imagesPath ):
    ( savedModel, savedImagesNumber, verSize, horSize, layersNum  ) = ANN.LoadANNModel(modelName)
    fileNames = os.listdir(imagesPath)
    for fn in fileNames:
        img = io.imread( imagesPath + "\\" + fn )
        ( ind, binInd ) = ANN.FindInANNModel( savedModel, img )
        print( "( ind, bin ) = ( " + str(ind) + "; " + str(binInd) + " ) ==>  fName = " + os.path.basename(fn) )

    # for i in range(fromRange, toRange):
    #     img = io.imread(imagesPath + "\img_" + str(i) + ".png")
    #     ind = find(img, savedModel)
    #     print("i = " + str(i) + ";   IND = " + str(ind))


if __name__ == "__main__":
    modelName = path + "\gfgModel.h5"


    # ------ создание
    print("\n *************     CreateModel       ***************  ")
    #CreateModel(path + "\IN", modelName, frVerSize, frHorSize )
    #
    #
    # # ----- использование
    print("\n *************     UseModel       ***************  ")
    #UseModel(modelName, 0, 255, path + "\OUT")

    # ------ реверс
    print("\n *************     ReverseModel       ***************  ")
    ReverseModel(modelName, path + "\OUT" )
