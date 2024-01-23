import os
import cv2
import json
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def imgs_to_matrix(imgs_dir, target_value):
    """
    Изображения представляет как матрицы значений пикселей.
    Присваивает изображениям логическую метку (True/False).

    :param imgs_dir:        путь до папки с изображениями.
    :param target_value:    логическая метка, которую необходимо присвоить всем изображениям в папке.

    :return:    imgs - матричные представления изображений (np.array).
                trgs - логические метки матричных представлений (np.array).
    """
    imgs, trgs = [], []

    for image in os.listdir(imgs_dir):

        if os.path.isfile(os.path.join(imgs_dir, image)):

            # читаем каждое изображение в оттенках серого
            image = cv2.imread(os.path.join(imgs_dir, image), cv2.IMREAD_GRAYSCALE)
            # изменяем масштаб на 64 х 64 рх
            image = cv2.resize(image, (64, 64))

            # сохраняем матрицу значений пикселей
            imgs.append(image)
            # и её логическую метку
            trgs.append(target_value)
            # индексы матриц и их логических меток совпадают

    return imgs, trgs


def compile_data():
    """
    Проходит циклом по директориям cars (изображения автомобилей) и no_cars (изображения без автомобилей).
    При помощи функции imgs_to_matrix конвертирует изображения в матрицы, присваивает им логические метки.

    :return: imgs_data - матричные представления изображений (np.array).
             trgs_data - логические метки матричных представлений (np.array).
    """
    # цикл по положительным изображениям
    cars = os.path.join(os.getcwd(), 'DB', 'STANFORD_IMG', 'cars')
    data_cars, target_cars = imgs_to_matrix(imgs_dir=cars, target_value=1)

    # цикл по отрицательным изображениям (то-же самое, но метка будет 0, а не 1)
    no_cars = os.path.join(os.getcwd(), 'DB', 'STANFORD_IMG', 'no_cars')
    data_no_cars, target_no_cars = imgs_to_matrix(imgs_dir=no_cars, target_value=0)

    imgs_data = data_cars + data_no_cars
    trgs_data = target_cars + target_no_cars

    # значения пикселей в матрицах конвертируем из действительного диапазона в вещественный (int => float)
    imgs_data = [d / 255. for d in imgs_data]

    # лист матриц конвертируем в np.array матриц (list[np.array] => np.array[np.array])
    imgs_data, trgs_data = np.array(imgs_data), np.array(trgs_data)

    return imgs_data, trgs_data


def save_compiled_data(imgs_np_array, trgs_np_array, name='DATA_DEFAULT'):
    """
    Сохраняет скомпилированные матричные представления изображений и их логические метки в файл JSON.

    :param imgs_np_array:   матричные представления изображений, созданные функцией imgs_to_matrix.
    :param trgs_np_array:   логические метки матричных представлений, созданные функцией imgs_to_matrix.
    :param name:            имя сохраняемого файла JSON.
    """
    # функция жрет память "как не в себя" - каждый символ в матрицах сохраняется в UTF-8

    imgs_list, trgs_list = imgs_np_array.tolist(), trgs_np_array.tolist()
    imgs_trgs = [img_and_trg for img_and_trg in zip(imgs_list, trgs_list)]

    path_to_file = os.path.join(os.getcwd(), 'DB', 'STANFORD_COMPILED_DATA', f'{name}.json')

    with open(file=path_to_file, mode='w', encoding='utf8') as data_file:
        json.dump(imgs_trgs, data_file, indent=2)


def load_compiled_data(name='DATA_DEFAULT'):
    """
    Читает файл JSON, созданный функцией save_compiled_data.
    Возвращает скомпилированные матричные представления изображений и их логические метки.

    :param name: имя файла JSON.

    :return: imgs_array - матричные представления изображений (np.array).
             trgs_array - логические метки матричных представлений (np.array).
    """
    path_to_file = os.path.join(os.getcwd(), 'DB', 'STANFORD_COMPILED_DATA', f'{name}.json')

    with open(file=path_to_file, mode='r', encoding='utf8') as data_file:
        imgs_trgs = json.load(data_file)

    # извлекаем из прочитанного файла матрицы значений пикселей и их логические метки
    imgs_list, trgs_list = [d[0] for d in imgs_trgs], [d[1] for d in imgs_trgs]
    # конвертируем списки матриц и их логических меток в np.appay
    imgs_array, trgs_array = np.array(imgs_list), np.array(trgs_list)

    return imgs_array, trgs_array


def get_torch_data_loaders(train_x, tests_x, train_y, tests_y, batch_size):
    """
    Получает скомпилированные функцией compile_data и разбитые на подгруппы данные.
    Возвращает загрузчики данных, необходимые для обучения модели в py_torch.


    :param train_x: набор матриц (матричных представлений изображений) для тренировки модели.
    :param tests_x: набор матриц (матричных представлений изображений) для тестирования модели.
    :param train_y: набор логических меток (правильных ответов) к матрицам для тренировки модели.
    :param tests_y: набор логических меток (правильных ответов) к матрицам для тестирования модели.
    :param batch_size: количество объектов, обрабатываемых моделью за одну итерацию.

    :return:    train_torch_loader - загрузчик данных для тренировки модели.
                tests_torch_loader - загрузчик данных для тестирования модели.
    """

    # конвертируем в tensor и переводим в 32-битный формат
    train_x = torch.from_numpy(train_x).to(torch.float32)
    train_y = torch.from_numpy(train_y).to(torch.int64)     # для бинарной классификации требуется LongTensor

    tests_x = torch.from_numpy(tests_x).to(torch.float32)
    tests_y = torch.from_numpy(tests_y).to(torch.int64)     # .type(torch.LongTensor) == .to(torch.int64)

    # создаем наборы данных (матрицы + метки)
    train_dataset = TensorDataset(train_x, train_y)
    tests_dataset = TensorDataset(tests_x, tests_y)

    # создаем загрузчики данных
    train_torch_loader = DataLoader(train_dataset, batch_size=batch_size)
    tests_torch_loader = DataLoader(tests_dataset, batch_size=batch_size)

    return train_torch_loader, tests_torch_loader
