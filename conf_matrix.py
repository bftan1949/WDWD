from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)
warnings.filterwarnings("ignore")


class TrainAndValidate:
    def __init__(self, X_encoded, y_label, test_id):
        self.X_encoded = X_encoded
        self.y_label = y_label

        self.update(test_id)

        self.classifier = SVC(class_weight={0: 2, 1: 4.5, 2: 1.5})

    def pred(self):
        self.classifier.fit(self.trian_X_encoded, self.trian_y_label)
        X = np.concatenate([self.trian_X_encoded, self.test_X_encoded])
        y = np.concatenate([self.trian_y_label, self.test_y_label], axis=0)
        predicts = self.classifier.predict(X=X)
        return predicts, y

    def update(self, test_id):
        self.test_id = test_id
        # remove 21 test data
        self.trian_X_encoded = np.delete(X_encoded, self.test_id, axis=0)
        self.trian_y_label = np.delete(y_label, self.test_id, axis=0)
        # shuffle the train data
        index = np.random.permutation(np.arange(len(self.trian_y_label)))
        self.trian_X_encoded = self.trian_X_encoded[index]
        self.trian_y_label = self.trian_y_label[index]
        # split the test data
        self.test_X_encoded = self.X_encoded[self.test_id]
        self.test_y_label = self.y_label[self.test_id]


#######################################################

data = pd.read_excel(r'data.xlsx', header=1)
index = list(data[~data['conductivity S/m'].isna()].index)
data = data.iloc[index]
data = data.sort_values(by='conductivity S/m')

attrs = ['ligands', 'ions', 'addition ratio', 'reaction duration/h', 'solvent']
target = ['conductivity S/m']

X = data[attrs]
y = data[target]
X_encoded = pd.get_dummies(X, columns=attrs)
y = np.asarray(y).flatten()

ranks = [1.21042931324091E-06, 0.0147100137185219]
y_label = np.digitize(y, bins=ranks)
X_encoded = np.asarray(X_encoded) + 0

new_values1 = np.zeros(shape=(2, X_encoded.shape[0]))
X_encoded = np.insert(X_encoded, 11, new_values1, axis=1)
new_values2 = np.zeros(shape=(3, X_encoded.shape[0]))
X_encoded = np.insert(X_encoded, 16, new_values2, axis=1)

class1 = [0, 11, 33, 34, 65, 100, 111]
class2 = [124, 127, 128, 129, 130, 132, 133]
class3 = [140, 180, 188, 198, 201, 204, 215]

tv = TrainAndValidate(X_encoded, y_label, class1 + class2 + class3)

preds, labels = tv.pred()
cm = confusion_matrix(labels, preds)
print(cm)

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['label 0', 'label 1', 'label 2'],
            yticklabels=['label 0', 'label 1', 'label 2'])
ax.set_xlabel('predicted labels')
ax.set_ylabel('true labels')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.savefig('./figs/conf_matrix.png', bbox_inches='tight')
