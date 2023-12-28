import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_binary_classification(names, prec_rec_values, prec_rec_auc, roc_values, roc_auc, acc_values, times):
  sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
  cmap = sns.color_palette('Paired')[1::2]
  fig, axs = plt.subplots(1, 4, figsize=(16, 5))
  for i in range(len(names)):
    axs[0].plot(roc_values[i][0], roc_values[i][1], label='{} AUC {:.3f}'.format(names[i], roc_auc[i]), c=cmap[i])
    axs[1].plot(prec_rec_values[i][1], prec_rec_values[i][0], label='{} AUC-PR {:.3f}'.format(names[i], prec_rec_auc[i]), c=cmap[i])
  axs[0].legend(loc='lower right')
  axs[0].set_xlabel('FPR')
  axs[0].set_ylabel('TPR')
  axs[0].set_title('ROC Curve')
  axs[1].legend(loc='lower left')
  axs[1].set_xlabel('Recall')
  axs[1].set_ylabel('Precision')
  axs[1].set_title('PR Curve')
  bars_acc = axs[2].bar(x=np.arange(len(names)), height=acc_values, color=cmap)
  axs[2].set_xticks(np.arange(len(names)))
  axs[2].set_xticklabels(names)
  axs[2].set_ylim()
  for bar_acc in bars_acc:
    yval = bar_acc.get_height()
    axs[2].text(bar_acc.get_x(), yval + .005, '{:.4f}'.format(yval))
  axs[2].set_ylabel('Accuracy')
  axs[2].set_title('Accuracy Bar Plots')
  bars = axs[3].bar(x=np.arange(len(names)), height=times, color=cmap)
  axs[3].set_xticks(np.arange(len(names)))
  axs[3].set_xticklabels(names)
  axs[3].set_ylim()
  for bar in bars:
    yval = bar.get_height()
    axs[3].text(bar.get_x(), yval + .005, '{:.4f}'.format(yval))
  axs[3].set_ylabel('Time Consumption in s')
  axs[3].set_title('Efficency')
  plt.tight_layout()
  plt.show()
