import numpy as np
from sklearn.linear_model import LinearRegression

from keras import ops


def normalize(data, mean, std):
    return (data - mean) / std


def create_dataset(
    label_compute_function,
    n_var=2,
    ranges=[-1, 1],
    train_num=1000,
    test_num=1000,
    normalize_input=False,
    normalize_label=False,
    seed=0,
):
    np.random.seed(seed)

    # Convert ranges to an array with shape (n_var, 2)
    ranges = np.array(ranges)
    if ranges.ndim == 1:
        ranges = np.tile(ranges, (n_var, 1))

    # Generate random samples within the specified ranges
    train_input = (
        np.random.rand(train_num, n_var) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    )
    test_input = (
        np.random.rand(test_num, n_var) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    )

    # Compute labels using the provided function f
    train_label = label_compute_function(train_input)
    test_label = label_compute_function(test_input)

    # Normalize input data if requested
    if normalize_input:
        mean_input = np.mean(train_input, axis=0, keepdims=True)
        std_input = np.std(train_input, axis=0, keepdims=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)

    # Normalize labels if requested
    if normalize_label:
        mean_label = np.mean(train_label, axis=0, keepdims=True)
        std_label = np.std(train_label, axis=0, keepdims=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    return train_input, train_label, test_input, test_label


def fit_params(
    x,
    y,
    symbolic_function,
    a_range=(-10, 10),
    b_range=(-10, 10),
    grid_number=101,
    iteration=3,
    verbose=True,
):
    for _ in range(iteration):
        a_ = ops.linspace(a_range[0], a_range[1], num=grid_number)
        b_ = ops.linspace(b_range[0], b_range[1], num=grid_number)
        a_grid, b_grid = ops.meshgrid(a_, b_, indexing="ij")
        post_fun = symbolic_function(
            a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :]
        )
        x_mean = ops.mean(post_fun, axis=[0], keepdims=True)
        y_mean = ops.mean(y, axis=[0], keepdims=True)
        numerator = (
            ops.sum((post_fun - x_mean) * (y - y_mean)[:, None, None], dim=0) ** 2
        )
        denominator = ops.sum((post_fun - x_mean) ** 2, axis=0) * ops.sum(
            (y - y_mean)[:, None, None] ** 2, axis=0
        )
        r2 = numerator / (denominator + 1e-4)
        r2 = ops.nan_to_num(r2)
        best_id = ops.argmax(r2)
        a_id = ops.floor_divide(best_id, grid_number)
        b_id = best_id % grid_number
        if (
            a_id != 0
            and a_id != grid_number - 1
            and b_id != 0
            and b_id != grid_number - 1
        ):
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]
    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = symbolic_function(a_best * x + b_best)
    r2_best = r2[a_id, b_id]
    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(
                f"r2 is not very high, please double check if you are choosing the correct symbolic function."
            )
    post_fun = ops.nan_to_num(post_fun)
    reg = LinearRegression().fit(
        ops.convert_to_numpy(post_fun[:, None]), ops.convert_to_numpy(y)
    )
    c_best = ops.convert_to_tensor(reg.coef_)[0]
    d_best = ops.convert_to_tensor(np.array(reg.intercept_))
    return ops.stack([a_best, b_best, c_best, d_best]), r2_best
