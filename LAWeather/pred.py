from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from xgboost import XGBClassifier


csv = pd.read_csv(r'./seattle-weather.csv')

csv = csv.drop(['date'], axis=1)

le = LabelEncoder()
csv['weather'] = le.fit_transform(csv['weather'])

x = np.array(csv.drop(['weather'], axis=1))
y = np.array(csv['weather'])
# xgb决策机
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    n_estimators=60,
    learning_rate=0.001,
    subsample=0.6,
)
xgb.fit(x_train, y_train)
# y_pred_xgb = xgb.predict(x_test)
# cfm = confusion_matrix(y_test, y_pred_xgb)
# disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=le.classes_, )
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()



