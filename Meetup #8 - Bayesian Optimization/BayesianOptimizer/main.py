from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# Target Function
def target(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ei", kappa=None, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('x', fontdict={'size': 20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.show()

x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)

plt.plot(x, y)
plt.show()

# Create a BayesianOptimization Object
optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=0)
optimizer.maximize(init_points=2, n_iter=0)

# Two random points
plot_gp(optimizer, x, y)

# After one-three step of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)

# After four steps of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)

# After five steps of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)

# After six steps of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)

# After seven steps of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)

# After eight steps of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)

# After nine steps of GP (and two random points)
optimizer.maximize(init_points=0, n_iter=1)
plot_gp(optimizer, x, y)



