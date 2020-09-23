# code for plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

adhoc_matrix = pd.read_csv("pyhanabi/tmp_data.csv", index_col=0)
ax = sns.heatmap(adhoc_matrix, annot=True, fmt=".1f")
plt.savefig('crossplay.png')