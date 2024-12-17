""" example of training a simple MLP model using flax
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import tqdm
import optax


def generate_sin_curve(phase, std, num_points):
    """generates a sin curve with some noise

    Args:
        phase (_type_): phase shift of the sin curve
        std (_type_): standard deviation of the noise
        num_points (_type_): number of points in the curve

    Returns:
        x, y=sin(x+phase)+noise: x and y values of the curve
    """
    x = jnp.linspace(0, 4 * jnp.pi, num_points)
    y = jnp.sin(x + phase) + std * random.normal(random.PRNGKey(0), (num_points,))
    return x, y


def plot_curve(x, y, ypred=None):
    """plots the curve

    Args:
        x (_type_): x
        y (_type_): y
        ypred (_type_, optional): y prediction. Defaults to None.
    
    Returns:
        plt: plot object
    """
    plt.plot(x, y, ".", alpha=0.5)
    if ypred is not None:
        plt.plot(x, ypred, "-")
    plt.xlabel("x")
    plt.ylabel("y")
    return plt

class MLP(nn.Module):
    """ simple MLP model
    
    Args:
        nn (_type_): linen module

    """
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


@jax.jit
def loss_fn(state, params, x, y):
    """ loss function
    
    Args:
        state: state of the model
        params: parameters of the model
        x (jnp.array): x
        y (jnp.array): y

    Returns:
        L2 loss
    """
    
    y_pred = state.apply_fn(params, x)
    return jnp.mean((y - y_pred) ** 2)


@jax.jit
def update_fn(state, x, y):
    """ update function
    
    Args:
        state (_type_): state of the model
        x (_type_): x
        y (_type_): y

    Returns:
        updated state of the model
    """
    loss, grad = jax.value_and_grad(loss_fn, argnums=1)(state, state.params, x, y)
    updates, new_opt_state = state.tx.update(grad, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(params=new_params, opt_state=new_opt_state), loss


if __name__ == "__main__":
    x, y = generate_sin_curve(0.0, 0.1, 1000)
    y = y.reshape(-1, 1)
    x = x.reshape(-1, 1)
    # plot_sin_curve(x[:, 0], y[:, 0])
    
    #(N, nparam)
    print(x.shape, y.shape)

    nfeatures = 32
    model = MLP(features=nfeatures)

    rng = random.PRNGKey(0)
    params = model.init(rng, x)
    # y_pred = model.apply(params, x)
    # plot_sin_curve(x[:, 0], y[:, 0], y_pred[:, 0])

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate=0.01),
    )

    loss_arr = []
    for _ in tqdm.tqdm(range(20000)):
        state, loss = update_fn(state, x, y)
        loss_arr.append(loss)  # loss should decrease

    y_pred = model.apply(state.params, x)
    plt = plot_curve(x[:, 0], y[:, 0], y_pred[:, 0])
    plt.savefig("mlp.png")