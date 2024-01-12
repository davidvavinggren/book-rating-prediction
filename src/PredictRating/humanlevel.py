from PredictRating.bertstats import plot_devs

from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

davvas_ratings = [3,5,4,1,2,3,1,3,2,5,1,1,2,3,1,4,5,4,4,5,3,5,2,5,4,2,1,3,4,2,5,2,4,2,5,3,2,4,2,4]
matildas_ratings = [3,5,3,4,1,3,1,4,3,5,1,1,2,3,1,4,5,4,3,5,3,5,3,5,3,1,1,3,4,2,4,1,4,2,5,4,2,5,3,3]
true_ratings = [3,4,3,4,1,2,1,3,2,4,1,1,3,3,1,4,5,4,2,5,2,5,2,5,4,1,1,3,3,2,5,1,4,2,5,5,3,4,2,5]

# Calc. stats for davva
cmat_davva = conf_mat(true_ratings, davvas_ratings)
disp = ConfusionMatrixDisplay(cmat_davva, display_labels = ['1', '2', '3', '4', '5'])
disp.plot()
plt.show()
acc_davva = cmat_davva.diagonal().sum() / cmat_davva.sum()
mse_davva = mean_squared_error(true_ratings, davvas_ratings)
print('acc davva:', acc_davva)
print('mse davva:', mse_davva)
plot_devs(true_ratings, davvas_ratings)
print(report(true_ratings, davvas_ratings))

# Calc. stats for matilda
cmat_matilda = conf_mat(true_ratings, matildas_ratings)
disp = ConfusionMatrixDisplay(cmat_matilda, display_labels = ['1', '2', '3', '4', '5'])
disp.plot()
plt.show()
acc_matilda = cmat_matilda.diagonal().sum() / cmat_matilda.sum()
print('acc matilda:', acc_matilda)
plot_devs(true_ratings, matildas_ratings)
print(report(true_ratings, davvas_ratings))

# Calc. cmat between davva and matilda
print(conf_mat(matildas_ratings, davvas_ratings))
disp = ConfusionMatrixDisplay(conf_mat(matildas_ratings, davvas_ratings), display_labels = ['1', '2', '3', '4', '5'])
disp.plot()
disp.ax_.set_xlabel('Predicted ratings by Subject 1')
disp.ax_.set_ylabel('Predicted ratings by Subject 2')
plt.show()
