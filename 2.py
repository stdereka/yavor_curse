import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
import numpy as np

data = pd.read_csv("2.csv")

x = data[["delta2e7"]].values
y = data[["I"]].values
xerr = data[["sigma2e7"]].values
yerr = data[["sigmai"]].values

print(st.pearsonr(x, y))

model = sm.OLS(y, sm.add_constant(x))
res = model.fit()
print(res.summary())

a = np.arange(min(x)-10, max(x)+10, 0.01)
plt.plot(a, -0.0410*a + 3.5257, color="k")
plt.plot(x, y, "bo")
plt.errorbar(x, y, xerr=xerr, yerr=yerr, ls="--", color="g", fmt=None, capsize=5)

xminorLocator = MultipleLocator(3)
yminorLocator = MultipleLocator(0.2)
xmajorLocator = MultipleLocator(15)
ymajorLocator = MultipleLocator(1)
ax = plt.subplot()
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_major_locator(xmajorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
plt.grid(which='major', ls='-', lw=0.5, c='k')
plt.grid(which='minor', ls='--', lw=0.5, c='grey')
plt.xlabel(u"$\Delta \Theta ^2,10^{-7}рад^2$")
plt.ylabel(u"$I^{(532)}, 50мВ$")
plt.xlim(0, 69)
plt.ylim(0, 4)

plt.show()
