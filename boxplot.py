import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

df = pd.read_csv('output.csv',sep = ';',decimal=',')

boxplot = df.boxplot(column=['Absolute Error MLP', 'Absolute Error SLR'])
plt.show()