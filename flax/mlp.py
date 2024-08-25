import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import tqdm
import optax

def generate_sin_curve(phase, std, num_points):
    x = jnp.linspace(0, 4 * jnp.pi, num_points)
    y = jnp.sin(x + phase) + std * random.normal(random.PRNGKey(0), (num_points,))
    return x, y


def plot_sin_curve(x, y, ypred=None):
    plt.plot(x, y, ".", alpha=0.5)
    if ypred is not None:
        plt.plot(x, ypred, "-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sin Curve")
    plt.show()


class MLP(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

@jax.jit
def loss_fn(params, x, y):
    y_pred = model.apply(params, x)
    return jnp.mean((y - y_pred) ** 2)

@jax.jit
def update_fn(state, x, y):
    loss, grad = jax.value_and_grad(loss_fn)(state.params, x, y)
    updates, new_opt_state = state.tx.update(grad, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(params=new_params, opt_state=new_opt_state), loss

if __name__ == "__main__":
    x, y = generate_sin_curve(0.0, 0.01, 100)
    y = y.reshape(-1, 1)
    x = x.reshape(-1, 1)
    #plot_sin_curve(x[:, 0], y[:, 0])

    nfeatures = 64
    model = MLP(features=nfeatures)
    
    rng = random.PRNGKey(0)
    initial_params = model.init(rng, x)
    y_pred = model.apply(initial_params, x)
    #plot_sin_curve(x[:, 0], y[:, 0], y_pred[:, 0])

    variables = model.init(rng, x)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optax.adam(learning_rate=0.01),
    )

    loss_arr = []
    for _ in tqdm.tqdm(range(50000)):
        state, loss = update_fn(state, x, y)
        loss_arr.append(loss)     # loss should decrease 
    
    y_pred = model.apply(state.params, x)
    plot_sin_curve(x[:, 0], y[:, 0], y_pred[:, 0])
