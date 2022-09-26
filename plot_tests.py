import json
import pprint
from util import plot_confusion_matrix, plot_residuals
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import tracemalloc
import numpy as np
import json
from sklearn.metrics import ConfusionMatrixDisplay

filename = "reports/number_metrics_7500_1500.json"
with open(filename) as f:
    data = json.load(f)
residuals_per_digit = {}
for i in range(10):
    residuals_per_digit[str(i)] = []
for id in data['results'].keys():
    k = str(data['results'][id]['actual'])
    residuals_per_digit[k].append(list(data['results'][id]['residuals']))
# every box:
# contains all residual values for one base (=one index in list)
boxplot_dict = {}
for i in range(10):
    i = str(i)
    boxplot_dict[i] = {}
    for k in range(10):
        boxplot_dict[i][k] = []
        for m in range(len(residuals_per_digit[i])):
            boxplot_dict[i][k].append(residuals_per_digit[i][m][k])
array_dict = {}
for i in range(10):
    array_dict[i] = [np.log2(inside_list) for inside_list in boxplot_dict[str(i)].values()]
del residuals_per_digit
del boxplot_dict

labels = [str(i) for i in range(10)]
labels.insert(0, "")
labels.append("")

fig, axs = plt.subplots(10,1,figsize=(15,100), dpi=300)
for i, ax in enumerate(axs.flat):
    ax.boxplot(array_dict[0])
    ax.set(xlabel='Basis', ylabel='$Log_{10}(residual)$')
    ax.set_title(f"Digit {str(i)}")
plt.setp(axs, xticks=list(range(12)), xticklabels=labels)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# plt.savefig(f"plots/residual_boxplots_{savestring}.pdf", format="pdf", bbox_inches="tight")
# plt.close(fig)
plt.show()



#ax.set_ylim(bottom=0, top=250)
# for k in str(range(10)):
#     for n in range(10):
#         for m in range(len(residuals_per_digit[k])):
#             residuals_per_digit[k][n][m]


# fig, axs = plt.subplots(10, 1, figsize=(15, 150), dpi=100)
# plt.suptitle("Residuals vs. Base for Every Actual Digit", fontsize=18, y=0.95)
# for i, ax in enumerate(axs.flat):
#     ax.boxplot(array_dict[i])
#     ax.set(xlabel='Basis', ylabel='Residual')
#     ax.set_title(f"Digit {str(i)}")
#     ax.label_outer()
# plt.setp(axs, xticks=list(range(10)), xticklabels=list(range(10)))
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# # plt.savefig(f"plots/boxplot_residual_.pdf", format="pdf", bbox_inches="tight")
# plt.show()
# # plt.close(fig)
