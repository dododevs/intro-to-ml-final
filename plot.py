import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_binary_classification(names, prec_rec_values, prec_rec_auc, roc_values, roc_auc, acc_values, times):
  sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
  cmap = sns.color_palette('colorblind', n_colors=len(names))
  fig, axs = plt.subplots(2, 2)#, figsize=(16, 5))
  fig.suptitle("Comparison of different learning techniques")
  for i in range(len(names)):
    axs[0, 0].plot(roc_values[i][0], roc_values[i][1], label='{} AUC {:.3f}'.format(names[i], roc_auc[i]), c=cmap[i])
    axs[0, 1].plot(prec_rec_values[i][1], prec_rec_values[i][0], label='{} AUC-PR {:.3f}'.format(names[i], prec_rec_auc[i]), c=cmap[i])
  axs[0, 0].legend(loc='lower right')
  axs[0, 0].set_xlabel('FPR')
  axs[0, 0].set_ylabel('TPR')
  axs[0, 0].set_title('ROC Curve')
  axs[0, 1].legend(loc='lower left')
  axs[0, 1].set_xlabel('Recall')
  axs[0, 1].set_ylabel('Precision')
  axs[0, 1].set_title('PR Curve')
  bars_acc = axs[1, 0].bar(x=np.arange(len(names)), height=acc_values, color=cmap)
  axs[1, 0].set_xticks(np.arange(len(names)))
  axs[1, 0].set_xticklabels(names)
  axs[1, 0].set_ylim()
  for bar_acc in bars_acc:
    yval = bar_acc.get_height()
    axs[1, 0].text(bar_acc.get_x(), yval + .005, '{:.4f}'.format(yval))
  axs[1, 0].set_ylabel('Accuracy')
  axs[1, 0].set_title('Accuracy Bar Plots')
  bars = axs[1, 1].bar(x=np.arange(len(names)), height=times, color=cmap)
  axs[1, 1].set_xticks(np.arange(len(names)))
  axs[1, 1].set_xticklabels(names)
  axs[1, 1].set_ylim()
  for bar in bars:
    yval = bar.get_height()
    axs[1, 1].text(bar.get_x(), yval + .005, '{:.4f}'.format(yval))
  axs[1, 1].set_ylabel('Time Consumption in s')
  axs[1, 1].set_title('Efficency')
  plt.tight_layout()
  plt.show()

  plt.figure()
  x = names
  y = roc_auc
  plt.bar(x,y,color=cmap)
  plt.title('ROC AUC comparison')
  plt.xlabel('model')
  plt.ylabel('AUC')
  plt.show()
