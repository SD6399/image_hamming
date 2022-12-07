from tensorflow import keras
import numpy as np
from keras.layers import Dense, Reshape
from keras.models import load_model
import os
from skimage import io
from keras.layers import ReLU
import tensorflow as tf
import math


def BuildANNModel(orig_folder, vert_size, hor_size, use_flip, use_rotate, extend_flag):
    bin_len=len_of_bin_vec(orig_folder, vert_size, hor_size, use_flip, use_rotate, extend_flag)
    print(bin_len)

    weight2 = deviding_by_img(orig_folder, vert_size, hor_size, use_flip, use_rotate,pow(2, bin_len))
    list_of_np1 = [weight2, np.zeros((hor_size * vert_size * 3))]
    weight_matrix = [list(bin(i)[2:]) for i in range(0, pow(2, bin_len))]
    for i in weight_matrix:
        while len(i) < len(weight_matrix[-1]):
            i.insert(0, 0)

    weight_matrix = np.array(weight_matrix, dtype=int)
    weight_matrix[weight_matrix == 0] = -1
    weight_matrix = np.rot90(weight_matrix)

    for j in range(len(weight_matrix[0])):
        weight_matrix[:, j] = weight_matrix[:, j][::-1]
    list_of_np = [weight_matrix, np.zeros(pow(2, bin_len))]

    model = keras.Sequential()

    model.add(Dense(pow(2, bin_len), activation=ReLU(threshold=bin_len-0.5),
                    input_shape=(bin_len,), name='layer_1'))

    model.add(Dense(vert_size * hor_size * 3, activation="relu"))
    model.add(Reshape((vert_size, hor_size, 3)))
    model.layers[0].set_weights(list_of_np)
    model.layers[1].set_weights(list_of_np1)
    return model


def len_of_bin_vec(orig_folder,vert_size,hor_size,use_flip,use_rotate,extend_flag):
    cnt=0
    pictures = os.listdir(orig_folder)
    for i in pictures:
        fname = orig_folder + "\\" + i
        pict = io.imread(fname)
        cnt += count_of_quadr_in_pic(pict, vert_size, hor_size)

    if use_flip:
        cnt *= 2
    if use_rotate:
        cnt *= 4
    if extend_flag:
        bin_len = int( math.ceil( math.log2( cnt ) ) )
    else:
        bin_len = int( math.floor( math.log2( cnt ) ) )

    return bin_len


def deviding_by_img(orig_folder, size_hor, size_vert, use_flip, use_rotate,count_neuron):
    pictures = os.listdir(orig_folder)
    weight = np.array([])
    for i in pictures:
        pict = io.imread(orig_folder + "\\" + i)
        for j in range(0, pict.shape[0]-size_hor+1, 1):
            for k in range(0, pict.shape[1]-size_vert+1, 1):
                if (pictures.index(i) == 0) and (j == k == 0):

                    orig_img = pict[0:0 + size_hor, 0:0 + size_vert]
                    weight = np.reshape(orig_img, (orig_img.shape[0] * orig_img.shape[1] * orig_img.shape[2]))
                    weight = weight[np.newaxis]
                else:
                    orig_img = np.zeros((size_hor, size_vert, 3))
                    tmp = pict[j:j + size_hor, k:k + size_vert]
                    orig_img[0:tmp.shape[0], 0: tmp.shape[1]] = tmp
                    cut_img_ax = np.reshape(orig_img, (orig_img.shape[0] * orig_img.shape[1] * orig_img.shape[2]))
                    cut_img_ax = cut_img_ax[np.newaxis]
                    if weight.shape[0] < count_neuron:
                        weight = np.append(weight, cut_img_ax, axis=0)
                        print(weight.shape)

                if use_rotate:
                    rotationNum = 4
                else:
                    rotationNum = 0

                for rot in range(1, rotationNum):
                    cut_img = np.rot90(orig_img,rot)
                    cut_img_ax = np.reshape(cut_img, (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                    cut_img_ax = cut_img_ax[np.newaxis]
                    if weight.shape[0] < count_neuron:
                        weight = np.append(weight, cut_img_ax, axis=0)

                    print(weight.shape)

                    if use_flip:
                        flip_img = cut_img[::-1, :, :]
                        flip_img = np.reshape(flip_img,
                                              (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                        flip_img = flip_img[np.newaxis]
                        if weight.shape[0] < count_neuron:
                            weight = np.append(weight, flip_img, axis=0)

    for i in range(weight.shape[0], count_neuron):
        cut_img = np.zeros((weight.shape[1]))
        cut_img = cut_img[np.newaxis]
        weight = np.append(weight, cut_img, axis=0)
        print(weight.shape)

    return weight


def count_of_quadr_in_pic(image, size_hor, size_vert):
    size_width = image.shape[0] - size_hor + 1
    size_high = image.shape[1] - size_vert + 1
    if (size_width < 0 ) or (size_high < 0):
        raise Exception("Negative number of pictures!")    #!!!!!!!!!!!!!!!!!!
    count = int(size_width * size_high)
    print(count)
    return count


def val_2_bin_list(N):
    return list(bin(N)[2:])


def val_2_tensor(model,N):
    max_len = np.log2(model.layers[0].output_shape[1])
    need_img= bin(N)
    inp = np.array(list(need_img.strip())[2:], dtype=int)
    while len(inp) < max_len:
        inp = np.insert(inp, 0, 0)
    inp[inp == 0] = -1
    X = tf.constant(np.reshape(inp, (1, len(inp))))
    return X


def FindInANNModel(model, image):

    vect_img= np.reshape(image,(image.shape[0]*image.shape[1]*image.shape[2]))
    weight= model.layers[1].get_weights()[0]
    for i in range(weight.shape[0]):
        tmp=weight[i,:]
        if np.array_equal(tmp, vect_img):
            return i, val_2_bin_list(i)


def ApplyANNModel( model, N, frVerSize, frHorSize ):
    X = val_2_tensor(model,N)
    Y = np.array(model(X))
    Y = np.reshape(Y, (frVerSize, frHorSize, 3))
    Y /= X.shape[1]

    return Y


def LoadANNModel( modelFileName ):
    savedModel = load_model(modelFileName,compile=False)

    savedImagesNumber = savedModel.weights[0].shape[1]
    verSize, horSize = savedModel.output_shape[1], savedModel.output_shape[2]

    layersNum=len(savedModel.weights)/2
    return savedModel, savedImagesNumber, verSize, horSize, layersNum

