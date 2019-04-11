import numpy as np
import ann as ann
import preprocessing as pp
import os as os
import tqdm as tqdm
import matplotlib.pyplot as plt


def classify(output, marks):
    errors = np.empty(len(marks))
    for i in range(len(marks)):
        errors[i] = np.linalg.norm(output - marks[i])
    return marks[errors.argmin()]


def act(x, a):
    return x


def dact(x, a):
    return a / np.cosh(a*x) / np.cosh(a*x)


if __name__ == '__main__':

    print_results_file, print_results_console = True, True
    standard_width, standard_height = 80, 60
    path_file_output = 'Вывод.txt'

    file_output = open(path_file_output, 'w')
    path_standard = 'Обучающая выборка/Стандарт/'
    path_original = 'Обучающая выборка/Оригинал/'
    a = 1.0
    expr_num = 15
    w_lower_bound, w_upper_bound = -0.95, 0.95
    class_number = 3
    epoch_n, learning_rate = 1000, 0.01
    folder_of_sample = 'К/', 'И/', 'В/', 'Нераспознанный символ/', 'Тест/'
    sample, sample_vol = [], 100
    test, test_vol = [], 30
    marks = tuple((e for e in np.eye(class_number)))
    tmp = []
    for cl in range(class_number):
        tmp.extend([marks[cl] for _ in range(test_vol // class_number)])
    test_output = np.array(tmp)
    sample_errors, test_errors = [], []
    next_phase_str = ''.join(['#' for _ in range(20)])
    prefix, postfix, new_line = '\n\n', '\n\n', '\n'
    figsize = (15.0, 7.5)

    msg = prefix + next_phase_str + '\nПредобработка исходной выборки\n' + next_phase_str + postfix
    if print_results_console:
        print(msg)
    if print_results_file:
        file_output.write(msg)
    pp.trim_edges(path_original, path_standard, folder_of_sample, folder_of_sample, True)
    pp.standardize_size(path_standard, path_standard, folder_of_sample, folder_of_sample,
                        standard_width, standard_height, True)

    msg = prefix + next_phase_str + '\nСоставление выборки\n' + next_phase_str + postfix
    if print_results_console:
        print(msg)
    if print_results_file:
        file_output.write(msg)
    for cl in range(class_number):
        examples_names = [file for file in os.listdir(path_standard + folder_of_sample[cl]) if file.endswith('.png')]
        for ex in tqdm.tqdm(examples_names):
            sample.append(pp.image_to_matrix(path_standard + folder_of_sample[cl] + ex).ravel())
    sample = np.array(sample)
    test_names = [file for file in os.listdir(path_standard + folder_of_sample[-1]) if file.endswith('.png')]
    # for t in tqdm.tqdm(test_names):
    #     test.append(pp.image_to_matrix(path_standard + folder_of_sample[-1] + t).ravel())
    for t in tqdm.tqdm(range(test_vol)):
        test.append(pp.image_to_matrix(path_standard + folder_of_sample[-1] + '%i.png' % t).ravel())
    test = np.array(test)
    output = np.vstack((marks[i] * np.ones((sample_vol, class_number)) for i in range(class_number)))

    neuron_n = np.array([class_number])
    for expr in range(expr_num):

        layer_n, neuron_in_each_n = expr + 1, 20
        msg = prefix + next_phase_str + '\nКоличество слоев нейронной сети равно %d\n' % layer_n + next_phase_str + \
              postfix
        if print_results_console:
            print(msg)
        if print_results_file:
            file_output.write(msg)
        print(neuron_n)
        pass
        pl = ann.PerceptronLgst(sample.shape[1], layer_n, neuron_n, learning_rate=learning_rate, epoch_n=epoch_n,
                                activation_param=a, w_lower_bound=w_lower_bound, w_upper_bound=w_upper_bound)

        msg = prefix + next_phase_str + '\nВывод нейронной сети с неоптимизированными весами\n' + next_phase_str + \
              postfix
        if print_results_console:
            print(msg)
        if print_results_file:
            file_output.write(msg)
        for i in range(sample.shape[0]):
            msg = '{0} - {1}{2}'.format(i, classify(pl.predict(sample[i]), marks), new_line)
            if print_results_console:
                print(msg)
            if print_results_file:
                file_output.write(msg)

        msg = prefix + next_phase_str + '\nОптимизация весов\n' + next_phase_str + postfix
        if print_results_console:
            print(msg)
        if print_results_file:
            file_output.write(msg)
        pl.fit(sample, output, test, test_output)

        msg = prefix + next_phase_str + '\nВывод нейронной сети с оптимизированными весами\n' + next_phase_str + \
              postfix
        if print_results_console:
            print(msg)
        if print_results_file:
            file_output.write(msg)
        for i in range(sample.shape[0]):
            msg = '{0} - {1}{2}'.format(i, classify(pl.predict(sample[i]), marks), new_line)
            if print_results_console:
                print(msg)
            if print_results_file:
                file_output.write(msg)

        msg = prefix + next_phase_str + '\nОбобщение нейронной сети на тестовой выборке\n' + next_phase_str + postfix
        if print_results_console:
            print(msg)
        if print_results_file:
            file_output.write(msg)
        for i in range(test.shape[0]):
            msg = '{0} - {1}{2}'.format(test_output[i], classify(pl.predict(test[i]), marks), new_line)
            if print_results_console:
                print(msg)
            if print_results_file:
                file_output.write(msg)

        plt.figure(expr, figsize=figsize)
        plt.title('Количество скрытых слоев равно %d' % expr)
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибки')
        plt.plot(np.arange(1, epoch_n + 1), pl.errors, 'r-', label='Ошибки обучения')
        plt.plot(np.arange(1, epoch_n + 1), pl.test_errors, 'b-', label='Ошибки обобщения')
        plt.legend()
        plt.grid(True)

        sample_errors.append(pl.errors.min())
        test_errors.append(pl.test_errors.min())

        # print(pl.test_errors.argmin())
        # print(pl.test_errors.min())
        # print(pl.errors[pl.test_errors.argmin()])
        neuron_n = np.concatenate((np.array([neuron_in_each_n]), neuron_n))
        # print(neuron_n)

    file_output.close()

    plt.figure(expr_num, figsize=figsize)
    plt.title('Зависимость минимальной ошибки обучения от количества скрытых слоев')
    plt.xlabel('Количество скрытых слоев')
    plt.ylabel('Минимальная ошибка обучения')
    plt.plot(np.arange(expr_num), sample_errors, 'k-')
    plt.grid(True)

    plt.figure(expr_num + 1, figsize=figsize)
    plt.title('Зависимость минимальной ошибки обобщения от количества скрытых слоев')
    plt.xlabel('Количество скрытых слоев')
    plt.ylabel('Минимальная ошибка обобщения')
    plt.plot(np.arange(expr_num), test_errors, 'k-')
    plt.grid(True)

    plt.show()
    plt.close()
