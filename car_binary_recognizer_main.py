import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as nn_f

from datetime import datetime
from sklearn.model_selection import train_test_split
from car_binary_recognizer_methods import compile_data, save_compiled_data, get_torch_data_loaders

BATCH = 64
EPOCHS = 4
LR = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEV_ID = torch.cuda.current_device()

SAVE_DATA = True
SAVE_MODEL = True
SAVE_MODEL_DICT = True
SAVE_MODEL_ONNX = True


class NeuralNetwork(nn.Module):
    """
    Класс нейронной сети. Три полностью связанных слоя. Функция потерь - leaky_relu.
    """
    def __init__(self):
        super().__init__()

        # fc => fully connected; 64 * 64 = 4096
        self.fcLayers = nn.ModuleList([
            nn.Linear(4096, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2),
        ])

    def forward(self, tensor):
        """
        :param tensor: матрица в формате 1х4096, представляющая пиксели ч.б. изображения в формате 64х64.
        :return: 1 - да, это автомобиль, 0 - нет, это не автомобиль.
        """
        for layer in self.fcLayers:
            tensor = nn_f.leaky_relu(layer(tensor), 0.1)
        return nn_f.log_softmax(tensor, dim=1)


def get_net_gpu(net=NeuralNetwork):
    """
    Создает экземпляр класса нейронной сети (модель). Если есть доступ к CUDA - передает обработку модели на GPU.

    :param net: класс нейронной сети.
    :return: model - экземпляр класса нейронной сети.
    """
    model = net()
    if DEVICE.__str__() == 'cuda':
        model = nn.DataParallel(model, device_ids=[DEV_ID])
        model = model.to(DEVICE)
    return model


def get_crit_gpu(crit=nn.CrossEntropyLoss):
    """
    Создает экземпляр класса обработчика обратного распространения ошибки.
    Если есть доступ к CUDA - передает обработку обратного распространения ошибки на GPU.

    :param crit: класс обработчика обратного распространения ошибки.
    :return: criterion - экземпляр класса обработчика обратного распространения ошибки.
    """
    criterion = crit()
    if DEVICE.__str__() == 'cuda':
        criterion = nn.DataParallel(criterion, device_ids=[DEV_ID])
        criterion = criterion.to(DEVICE)
    return criterion


if __name__ == '__main__':

    # представляем изображения в виде матриц
    data, target = compile_data()

    # сохраняем матричные представления изображений
    if SAVE_DATA:
        save_compiled_data(data, target, f'NEURA_{EPOCHS}')     # жрет память по богатырски :)

    # T_X, V_X — матрицы данных для тренировки и тестирования модели
    # t_y, v_y — логические метки матриц (правильные ответы)
    T_X, V_X, t_y, v_y = train_test_split(data, target, train_size=0.75, random_state=42, shuffle=True)

    # создаем загрузчики данных
    train_loader, tests_loader = get_torch_data_loaders(T_X, V_X, t_y, v_y, BATCH)

    # создаем экземпляр нейросети
    neura = get_net_gpu(net=NeuralNetwork)

    # создаем функцию потерь
    neura_criterion = get_crit_gpu(crit=nn.CrossEntropyLoss)

    # создаем оптимизатор потерь
    neura_optimizer = optim.Adam(neura.parameters(), lr=LR)

    # включили режим обучения модели
    neura.train()

    for epoch in range(EPOCHS):
        print(f'Epoch №{epoch + 1}, time: {datetime.now().strftime("%H:%M:%S")}')
        for data, target in train_loader:
            neura_optimizer.zero_grad()

            result = neura(data.view(-1, 64 * 64))
            losses = neura_criterion(input=result, target=target)

            losses.backward()       # обратное распространение ошибки
            neura_optimizer.step()  # корректировка весов с учетом lr

    # выключили режим обучения модели
    neura.eval()

    # тестируем обученную модель
    right, wrong = 0, 0

    with torch.no_grad():
        for data, target in tests_loader:

            result = neura(data.view(-1, 64 * 64))

            for idx, res in enumerate(result):
                right += 1 if torch.argmax(res) == target[idx] else 0
                wrong += 1 if torch.argmax(res) != target[idx] else 0

    # сохраняем обученную модель
    if SAVE_MODEL:
        torch.save(neura,
                   f=f'DB\\STANFORD_SAVED_MODEL\\NEURA_{EPOCHS}.pth')

    # сохраняем model_dict для дальнейшего обучения модели с текущего состояния
    if SAVE_MODEL_DICT:
        torch.save(neura.state_dict(),
                   f=f'DB\\STANFORD_SAVED_MODEL_DICTS\\NEURA_DICT_{EPOCHS}.pth')

    # сохраняем обученную модель в переходном формате onnx (сериализация модели)
    if SAVE_MODEL_ONNX:
        if not SAVE_MODEL:
            print('Для сохранения модели в ONNX должен быть включен параметр SAVE_MODEL')
        else:
            DEVICE = torch.device('cpu')

            neura = torch.load(f'DB\\STANFORD_SAVED_MODEL\\NEURA_{EPOCHS}.pth').to(DEVICE)

            torch.onnx.export(neura.module,
                              torch.rand(1, 64 * 64),
                              f=f'DB\\STANFORD_SAVED_ONNX\\NEURA_{EPOCHS}.onnx')

    print(f'Accuracy: {100 * right / (right + wrong)}.\n'
          f'Test cases passed: {(right + wrong)}.')
