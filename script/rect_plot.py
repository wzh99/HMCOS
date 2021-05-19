import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (6, 6)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.sans-serif'] = 'Segoe UI'

ax = plt.gca()
ax.add_patch(plt.Rectangle((2, 2), 3, 3, facecolor='dodgerblue'))
ax.add_patch(plt.Rectangle((2, 7), 2, 2, facecolor='mediumslateblue'))
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
