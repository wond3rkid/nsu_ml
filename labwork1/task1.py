import numpy as np
import matplotlib.pyplot as plt

N = 100
epsilons_0 = [0.1, 0.6]
fig, axes = plt.subplots(4, 2, figsize=(15, 15))


def generate_sample(func, error_type: str, sample_size: int, epsilon_0: float) -> [float, float]:
    """
    Генерация выборки с заданным типом ошибки
    :param func: функция, по которой генерируется выборка
    :param error_type: тип ошибки
    :param sample_size: размер выборки
    :param epsilon_0: граница интервала ошибки
    :return: выборка
    """
    x = np.random.uniform(-1, 1, sample_size)
    y_true = func(x)

    if error_type == 'uniform':
        epsilon = np.random.uniform(-epsilon_0, epsilon_0, sample_size)
    elif error_type == 'normal':
        sigma = epsilon_0 / 3
        epsilon = np.clip(np.random.normal(0, sigma, sample_size), -epsilon_0, epsilon_0)
    else:
        print(f'U entered wrong error type: {error_type}. Try again!')
        raise ValueError("Некорректный тип ошибки")

    return x, y_true + epsilon


def generate_cubic_sample():
    """
    Генерация кубической выборки
    """
    np.random.seed(40)
    a, b, c, d = np.random.uniform(-3, 3, 4)
    cubic_func = lambda var: a * var ** 3 + b * var ** 2 + c * var + d
    for i, eps in enumerate(epsilons_0):
        x, y = generate_sample(func=cubic_func, error_type='uniform', sample_size=N, epsilon_0=eps)
        x_vals = np.linspace(-1, 1, 200)
        axes[0, i].plot(x_vals, cubic_func(x_vals), 'b-', label='True function')
        axes[0, i].scatter(x, y, c='r', s=20, label='Samples')
        axes[0, i].set_title(f'Cubic func: Uniform error ε={eps}')
        axes[0, i].grid(True)

        x, y = generate_sample(func=cubic_func, error_type='normal', sample_size=N, epsilon_0=eps)
        x_vals = np.linspace(-1, 1, 200)
        axes[1, i].plot(x_vals, cubic_func(x_vals), 'b-', label='True function')
        axes[1, i].scatter(x, y, c='r', s=20, label='Samples')
        axes[1, i].set_title(f'Cubic func: Normal error ε={eps}')
        axes[1, i].grid(True)


def generate_sinus_sample():
    """
    Генерация синусоидальной выборки
    """
    sin_func = lambda var: np.sin(2 * np.pi * var)

    for i, eps in enumerate(epsilons_0):
        x, y = generate_sample(func=sin_func, error_type='uniform', sample_size=N, epsilon_0=eps)
        x_vals = np.linspace(-1, 1, 200)
        axes[2, i].plot(x_vals, sin_func(x_vals), 'b-', label='True function')
        axes[2, i].scatter(x, y, c='r', s=20, label='Samples')
        axes[2, i].set_title(f'Sin func: Uniform error ε={eps}')
        axes[2, i].grid(True)

        x, y = generate_sample(func=sin_func, error_type='normal', sample_size=N, epsilon_0=eps)
        x_vals = np.linspace(-1, 1, 200)
        axes[3, i].plot(x_vals, sin_func(x_vals), 'b-', label='True function')
        axes[3, i].scatter(x, y, c='r', s=20, label='Samples')
        axes[3, i].set_title(f'Sin func: Normal error ε={eps}')
        axes[3, i].grid(True)


def make_task():
    generate_cubic_sample()
    generate_sinus_sample()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    make_task()
