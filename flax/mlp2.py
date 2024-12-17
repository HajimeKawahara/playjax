""" example of training a simple MLP model using flax
    sin(2*pi*x/P + phi)
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import numpy as np
import tqdm
import optax


def generate_sin_curve(period, phase, std, num_points):
    """generates a sin curve with some noise

    Args:
        period (_type_): period of the sin curve
        phase (_type_): phase shift of the sin curve
        std (_type_): standard deviation of the noise
        num_points (_type_): number of points in the curve

    Returns:
        x, y=sin(x+phase)+noise: x and y values of the curve
    """
    x = jnp.linspace(0, 4 * jnp.pi, num_points)
    y = jnp.sin(2.0*jnp.pi*x[:,np.newaxis]/period[np.newaxis,:] + phase[np.newaxis,:]) + std * random.normal(random.PRNGKey(0), (num_points,len(period)))
    xtile = jnp.tile(x, (len(period),1)).T
    return xtile, y


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

class MLP_parallel(nn.Module):
    """ simple MLP model
    
    Args:
        nn (_type_): linen module

    """
    features: int

    @nn.compact
    def __call__(self, x, p, q):
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

        p = jnp.ones_like(x) * p
        p = jnp.expand_dims(p, axis=1)
        p = nn.Dense(features=self.features)(p)
        p = nn.relu(p)
        p = nn.Dense(features=self.features)(p)
        p = nn.relu(p)
        p = nn.Dense(features=self.features)(p)
        p = nn.relu(p)
        p = nn.Dense(features=self.features)(p)
        p = nn.relu(p)
        p = nn.Dense(features=self.features)(p)
        p = nn.relu(p)

        q = jnp.ones_like(x) * q
        q = jnp.expand_dims(q, axis=1)
        q = nn.Dense(features=self.features)(q)
        q = nn.relu(q)
        q = nn.Dense(features=self.features)(q)
        q = nn.relu(q)
        q = nn.Dense(features=self.features)(q)
        q = nn.relu(q)
        q = nn.Dense(features=self.features)(q)
        q = nn.relu(q)
        q = nn.Dense(features=self.features)(q)
        q = nn.relu(q)

        x = jnp.concatenate([x, p, q], axis=1)
        x = nn.Dense(features=1)(x)


        return x

class MLP_conc(nn.Module):
    """ simple MLP model
    
    Args:
        nn (_type_): linen module

    """
    features: int

    @nn.compact
    def __call__(self, x, p, q):
        xcon = jnp.concatenate([x, p[jnp.newaxis,:], q[jnp.newaxis,:]], axis=0)
        xcon = nn.Dense(features=self.features)(xcon)
        xcon = nn.relu(xcon)
        xcon = nn.Dense(features=self.features)(xcon)
        xcon = nn.relu(xcon)
        xcon = nn.Dense(features=self.features)(xcon)
        xcon = nn.relu(xcon)
        xcon = nn.Dense(features=self.features)(xcon)
        xcon = nn.relu(xcon)
        xcon = nn.Dense(features=self.features)(xcon)
        xcon = nn.relu(xcon)
        x = nn.Dense(features=len(p))(xcon)

        return x


@jax.jit
def loss_fn(state, params, x, p, q, y):
    """ loss function
    
    Args:
        state: state of the model
        params: parameters of the model
        x (jnp.array): x
        y (jnp.array): y

    Returns:
        L2 loss
    """
    
    y_pred = state.apply_fn(params, x, p, q)
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
    num_samples = 100
    p_train = np.random.uniform(0.1, 5.0, num_samples)  # Random values for P in [0.1, 5.0]
    q_train = np.random.uniform(-jnp.pi, jnp.pi, num_samples) 
    xnum_samples = 1000
    x, y = generate_sin_curve(period=p_train, phase=q_train, std=0.1, num_points=xnum_samples)
    
    print(x.shape, y.shape)
    
    #y = y.reshape(-1, 1)
    #x = x.reshape(-1, 1)
    
    
    # plot_sin_curve(x[:, 0], y[:, 0])

    nfeatures = 32
    model = MLP_conc(features=nfeatures)

    rng = random.PRNGKey(0)
    params = model.init(rng, x, p_train, q_train)
    y_pred = model.apply(params, x, p_train, q_train)
    print(np.shape(x),np.shape(y_pred),np.shape(y))
    
    #plot_curve(x[:1000, 0], y[:, 0], y_pred[:, 0])

    import sys
    sys.exit(0)

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