from scipy.stats import weibull_min

def weibull_dist(lam, k):
    return weibull_min(k, scale=lam)

lam = 3
k = 0.8
actual_dist = weibull_dist(lam, k)

import numpy as np
from empiricaldist import Cdf



qs = np.linspace(0, 12, 101)
ps = actual_dist.cdf(qs)
cdf = Cdf(ps, qs)
cdf.plot()

def decorate():
    xlabel='Duration in time'
    ylabel='CDF'
    title='CDF of a Weibull distribution'

data = actual_dist.rvs(10)
print(data)

from utils import make_uniform

lams = np.linspace(0.1, 10.1, num=101)
prior_lam = make_uniform(lams, name='lambda')

ks = np.linspace(0.1, 5.1, num=101)
prior_k = make_uniform(ks, name='k')

from utils import make_joint

prior = make_joint(prior_lam, prior_k)

lam_mesh, k_mesh, data_mesh = np.meshgrid(
    prior.columns, prior.index, data)

densities = weibull_dist(lam_mesh, k_mesh).pdf(data_mesh)
print(densities.shape)