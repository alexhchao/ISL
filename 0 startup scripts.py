import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/49918998/plt-show-not-working-in-pycharm

#import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
pd.options.display.float_format = '{:,.4f}'.format
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
