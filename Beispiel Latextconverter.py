import numpy as np
import tabulate as tl
x_data = np.array([1,2,3,4,5,6,7,8,9,10])
y_data = x_data*2
table = np.array([x_data, y_data]).transpose()
print(tl.tabulate(table, headers=["x","y"], tablefmt = "latex_booktabs", numalign="center", stralign="center"))