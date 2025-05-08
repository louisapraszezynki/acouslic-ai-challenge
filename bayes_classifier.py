from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# conf_mat = confusion_matrix(y_true, y_pred)
#
# print(conf_mat)

disp = ConfusionMatrixDisplay(confusion_matrix=np.array([
    [0.78, 0.17, 0.04],
    [0.03, 0.96, 0],
    [0.04, 0.007, 0.94]
]))
disp.plot()
plt.show()