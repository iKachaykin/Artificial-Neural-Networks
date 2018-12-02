from PIL import Image
import tqdm as tqdm
import os as os
import imageio as imageio
import numpy as np


def image_to_matrix(path):
    image = imageio.imread(path)
    return np.where(np.array(np.all(image == 255, 2), dtype=int) == 1, 0, 1)


def matrix_to_image(path, matrix):
    inds0, inds1 = np.where(matrix == 1), np.where(matrix == 0)
    tmp_res = np.empty((600, 800, 4))
    tmp_res[inds0[0], inds0[1], :] = np.array([255, 255, 255, 255], dtype=np.uint8)
    tmp_res[inds1[0], inds1[1], :] = np.array([0, 0, 0, 255], dtype=np.uint8)
    tmp_res = np.array(tmp_res, dtype=np.uint8)
    imageio.imwrite(path, tmp_res)


def standardize_size(path_source, path_dest, subfolders_source, subfolders_dest, new_width, new_height, tqdm_flag=False):
    if len(subfolders_source) != len(subfolders_dest):
        raise ValueError('Sizes of arguments "subfolders_source" and "subfolders_dest" must correspond to each other!')
    for sf in range(len(subfolders_source)):
        old_imgs_names = [file for file in os.listdir(path_source + subfolders_source[sf]) if file.endswith('.png')]
        name_it = tqdm.tqdm(old_imgs_names) if tqdm_flag else old_imgs_names
        for name in name_it:
            old_img = Image.open(path_source + subfolders_source[sf] + name)
            new_img = old_img.resize((new_width, new_height))
            new_img.save(path_dest + subfolders_dest[sf] + name)


def trim_edges(path_source, path_dest, subfolders_source, subfolders_dest, tqdm_flag=False, margin=(0, 0, 0, 0)):
    if len(subfolders_source) != len(subfolders_dest):
        raise ValueError('Sizes of arguments "subfolders_source" and "subfolders_dest" must correspond to each other!')
    m_left, m_right, m_top, m_bot = margin
    for sf in range(len(subfolders_source)):
        old_imgs_names = [file for file in os.listdir(path_source + subfolders_source[sf]) if file.endswith('.png')]
        name_it = tqdm.tqdm(old_imgs_names) if tqdm_flag else old_imgs_names
        for name in name_it:
            old_img = imageio.imread(path_source + subfolders_source[sf] + name)
            old_img_matrix = np.where(np.array(np.all(old_img == 255, 2), dtype=int) == 1, 0, 1)
            tmp_matrix = np.argwhere(old_img_matrix)
            i_min, j_min = tuple(tmp_matrix[tmp_matrix.argmin(axis=0)])
            i_max, j_max = tuple(tmp_matrix[tmp_matrix.argmax(axis=0)])
            i_min, j_min = i_min[0] - m_top if i_min[0] - m_top >= 0 \
                               else 0,\
                           j_min[1] - m_left if j_min[1] - m_left >= 0 else 0
            i_max, j_max = \
                i_max[0] + m_bot if i_max[0] + m_bot < old_img.shape[0] else old_img.shape[0] - 1,\
                j_max[1] + m_right if j_max[1] + m_right < old_img.shape[1] else old_img.shape[1] - 1
            imageio.imwrite(path_dest + subfolders_source[sf] + name, old_img[i_min:i_max+1, j_min:j_max+1])


if __name__ == '__main__':
    new_width, new_height = 80, 60
    path_dest = '/Users/ivankachaikin/Documents/Учеба/Нейронные сети и машинное обучение/Лабораторные работы/Л. р. 1/' \
           'Обучающая выборка/Стандарт/'
    path_source = '/Users/ivankachaikin/Documents/Учеба/Нейронные сети и машинное обучение/Лабораторные работы/Л. р. 1/' \
           'Обучающая выборка/Оригинал/'
    subfolders = 'К/', 'В/', 'И/', 'Нераспознанный символ/', 'Тест/'
    trim_edges(path_source, path_dest, subfolders, subfolders, True)
    # for sf in range(len(subfolders)):
    #     for i in tqdm.tqdm(range(sample_vol[sf])):
    #         old_img = Image.open(path_source + subfolders[sf] + '%i.png' % i)
    #         new_img = old_img.resize((new_width, new_height))
    #         new_img.save(path_dest + subfolders[sf] + '%i.png' % i)

