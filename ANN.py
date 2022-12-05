from tensorflow import keras
import numpy as np
from keras.layers import Dense, Reshape
import os
from skimage import io
from keras.layers import ReLU
import tensorflow as tf


def build_ANN(orig_folder, vert_size, hor_size, use_flip, use_rotate, extend_flag):
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
        pict = io.imread(orig_folder + i)
        cnt += count_of_quadr_in_pic(pict, vert_size, hor_size)

    cnt -= 1

    if use_flip and use_rotate:
        bin_len = len(val_2_bin_list(cnt*8))
    elif use_flip and use_rotate==False:
        bin_len = len(val_2_bin_list(cnt * 2))
    elif use_rotate==True and use_flip==False:
        bin_len = len(val_2_bin_list(cnt * 4))
    else:
        bin_len = len(val_2_bin_list(cnt))
    if not extend_flag:
        bin_len-=1

    return bin_len


def deviding_by_img(orig_folder, size_hor, size_vert, use_flip, use_rotate,count_neuron):
    pictures = os.listdir(orig_folder)
    weight = np.array([])
    for i in pictures:
        pict = io.imread(orig_folder + i)
        for j in range(0, pict.shape[0]-size_hor+1, size_hor):
            for k in range(0, pict.shape[1]-size_vert+1, size_vert):
                if (pictures.index(i) == 0) and (j == k == 0):

                    cut_img = pict[0:0 + size_hor, 0:0 + size_vert]
                    weight = np.reshape(cut_img, (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                    weight = weight[np.newaxis]
                    if use_flip:
                        flip_img = cut_img[::-1, :, :]
                        flip_img = np.reshape(flip_img, (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                        flip_img = flip_img[np.newaxis]

                        weight = np.append(weight, flip_img, axis=0)
                    if use_rotate:
                        for rot in range(90, 360, 90):
                            cut_img = np.rot90(cut_img)
                            cut_img_ax = np.reshape(cut_img, (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                            cut_img_ax = cut_img_ax[np.newaxis]

                            weight = np.append(weight, cut_img_ax, axis=0)

                            print(weight.shape)
                            if use_flip:
                                flip_img = cut_img[::-1, :, :]
                                flip_img = np.reshape(flip_img,
                                                      (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                                flip_img = flip_img[np.newaxis]

                                weight = np.append(weight, flip_img, axis=0)

                                print(weight.shape)

                else:

                    orig_img = np.zeros((size_hor, size_vert, 3))
                    tmp = pict[j:j + size_hor, k:k + size_vert]
                    orig_img[0:tmp.shape[0], 0: tmp.shape[1]] = tmp
                    cut_img_ax = np.reshape(orig_img, (orig_img.shape[0] * orig_img.shape[1] * orig_img.shape[2]))
                    cut_img_ax = cut_img_ax[np.newaxis]

                    weight = np.append(weight, cut_img_ax, axis=0)
                    cut_img= orig_img
                    if use_rotate:
                        for rot in range(90, 360, 90):
                            cut_img = np.rot90(cut_img)
                            cut_img_ax = np.reshape(cut_img, (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                            cut_img_ax = cut_img_ax[np.newaxis]

                            weight = np.append(weight, cut_img_ax, axis=0)

                            print(weight.shape)

                            if use_flip:
                                flip_img = cut_img[::-1, :, :]
                                flip_img = np.reshape(flip_img,
                                                      (cut_img.shape[0] * cut_img.shape[1] * cut_img.shape[2]))
                                flip_img = flip_img[np.newaxis]

                                weight = np.append(weight, flip_img, axis=0)
                    if use_flip:
                        flip_img = orig_img[::-1, :, :]
                        flip_img = np.reshape(flip_img,
                                              (orig_img.shape[0] * orig_img.shape[1] * orig_img.shape[2]))
                        flip_img = flip_img[np.newaxis]

                        weight = np.append(weight, flip_img, axis=0)

                        print(weight.shape)

    for i in range(weight.shape[0], count_neuron):
        cut_img = np.zeros((weight.shape[1]))
        cut_img = cut_img[np.newaxis]
        weight = np.append(weight, cut_img, axis=0)
        print(weight.shape)

    return weight


def count_of_quadr_in_pic(image, size_hor, size_vert):
    size_width = np.floor(image.shape[0] / size_hor)
    size_high = np.floor(image.shape[1] / size_vert)
    count = int(size_width * size_high)
    return count


def val_2_bin_list(N):
    return list(bin(N)[2:])


def val_2_tensor(N,model):
    max_len = np.log2(model.layers[0].output_shape[1])
    need_img= bin(N)
    inp = np.array(list(need_img.strip())[2:], dtype=int)
    while len(inp) < max_len:
        inp = np.insert(inp, 0, 0)
    inp[inp == 0] = -1
    X = tf.constant(np.reshape(inp, (1, len(inp))))
    return X


def find(image,model):

    vect_img= np.reshape(image,(image.shape[0]*image.shape[1]*image.shape[2]))

    weight= model.layers[1].get_weights()[0]
    for i in range(weight.shape[0]):
        tmp=weight[i,:]
        if np.array_equal(tmp, vect_img):
            return val_2_bin_list(i)


def apply(N,model):
    X = val_2_tensor(N, model)
    Y = np.array(model(X))
    Y = np.reshape(Y, (128, 128, 3))
    Y /= X.shape[1]

    return Y

