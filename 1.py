import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
import numpy as np

data = pd.read_csv("1.csv")
first = data["first"].values
first2 = data["first2"].values
second = data["second"].values
err = data["sigma2"].values

print(st.pearsonr(first2, second))

model = sm.OLS(second, first2)
res = model.fit()
print(res.summary())

a = np.arange(min(first2)-10, max(first2)+10, 0.01)
plt.plot(a, 0.1918*a, color="k")
plt.plot(first2, second, "bo")
plt.errorbar(first2, second, xerr=0.1, yerr=err, ls="--", color="g", fmt=None, capsize=5)

xminorLocator = MultipleLocator(0.5)
yminorLocator = MultipleLocator(0.2)
xmajorLocator = MultipleLocator(2.5)
ymajorLocator = MultipleLocator(1)
ax = plt.subplot()
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_major_locator(xmajorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
plt.grid(which='major', ls='-', lw=0.5, c='k')
plt.grid(which='minor', ls='--', lw=0.5, c='grey')
plt.xlabel(u"$(I^{(1064)})^2,(2В)^2$")
plt.ylabel(u"$I^{532},50мВ$")
plt.xlim(0, 21)
plt.ylim(0, 5)

plt.show()
