import numpy as np
import matplotlib.pyplot as plt

sample_size: int = 100
func = lambda x: x * np.sin(2 * np.pi * x)
noise_scale: float = 0.25


def generate_sample() -> tuple:
    """Генерация зашумлённой выборки"""
    x = np.random.uniform(-1, 1, sample_size)
    y = func(x) + np.random.normal(0, noise_scale, sample_size)
    return x, y


def polynomial_regression(x, y, degree: int) -> np.ndarray:
    """Полиномиальная регрессия"""
    x = np.vander(x, degree + 1, increasing=True)
    return np.linalg.inv(x.T @ x) @ x.T @ y


def predict(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Прогнозирование значений полинома"""
    return np.sum([c * x ** i for i, c in enumerate(coeffs)], axis=0)


def plot_results(ax, x_train: np.ndarray, y_train: np.ndarray, coeffs: np.ndarray, degree: int, title: str) -> None:
    """Визуализация на указанных осях"""
    x_plot = np.linspace(-1, 1, 200)
    y_true = func(x_plot)
    y_pred = predict(x_plot, coeffs)

    ax.scatter(x_train, y_train, s=20, c='red', label='Выборка')
    ax.plot(x_plot, y_true, 'b-', lw=2, label='Истинная функция')
    ax.plot(x_plot, y_pred, 'g--', lw=2, label=f'Степень {degree}')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def make_task():
    """Отрисовка всех графиков"""
    x, y = generate_sample()

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    degrees = [1, 20, 8]
    degree_1 = f'Недообучение (degree={degrees[0]})'
    degree_2 = f'Переобучение (degree={degrees[1]})'
    degree_3 = f'Оптимальная модель (degree={degrees[2]})'

    titles = [degree_1, degree_2, degree_3]

    for i, (degree, title) in enumerate(zip(degrees, titles)):
        row, col = divmod(i, 2)
        coeffs = polynomial_regression(x, y, degree)
        plot_results(axs[row, col], x, y, coeffs, degree, title)

    x_plot = np.linspace(-1, 1, 200)
    axs[1, 1].scatter(x, y, s=20, c='red', label='Выборка')
    axs[1, 1].plot(x_plot, func(x_plot), 'b-', lw=2, label='Истинная функция')
    for degree, style in zip(degrees, ['--', ':', '-.']):
        coeffs = polynomial_regression(x, y, degree)
        y_pred = predict(x_plot, coeffs)
        axs[1, 1].plot(x_plot, y_pred, style, lw=2, label=f'Степень {degree}')
    axs[1, 1].set_title('Все модели')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    make_task()
