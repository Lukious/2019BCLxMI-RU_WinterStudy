#%%
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns

if __name__=='__main__':
    iris = datasets.load_iris()
    print('아이리스 종류 : ', iris.target_names)
    print('target : [0:setosa, 1:versicolor, 2:virginica]')
    print('데이터 수 : ', len(iris.data))
    print('데이터 열 이름 : ', iris.feature_names)
    
    # iris data Dataframe으로
    data = pd.DataFrame(
        {
            'sepal length':iris.data[:,0],
            'sepal width':iris.data[:,1],
            'petal length':iris.data[:,2],
            'petal width':iris.data[:,3],
            'species':np.array([iris.target_names[i] for i in iris.target])
            }
        )
    
sns.pairplot(data, hue='species')

#%%
from sklearn.model_selection import train_test_split

x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['species']

# 테스트 데이터 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# 학습 진행
forest = RandomForestClassifier(n_estimators=5, n_jobs=-1)
forest.fit(x_train, y_train)

# 예측
y_pred = forest.predict(x_test)
print(y_pred)
print(list(y_test))

# 정확도 확인
print('정확도:', metrics.accuracy_score(y_test, y_pred))

#%%
import matplotlib.pyplot as plt
from mglearn.plots import plot_2d_classification

_,axes = plt.subplots(2,3)
marker_set = ['o','x','^']

# for i, (axe, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#     axe.set_title('tree {}'.format(i))
#     plot_2d_classification(tree, x, fill=True, ax=axe, alpha=0.4)

#     for i, m in zip(np.unique(y), marker_set):
#         axe.scatter(x[y==i][:, 0], x[y==i][:, 1], marker=m,
#                     label='class {}'.format(i), edgecolors='k')
#         axe.set_xlabel('feature 0')
#         axe.set_ylabel('feature 1')
        
axes[-1, -1].set_title('random forest')
axes[-1, -1].set_xlabel('feature 0')
axes[-1, -1].set_ylabel('feature 1')
plot_2d_classification(forest, x, fill=True, ax=axes[-1, -1], alpha=0.4)

for i, m in zip(np.unique(y), marker_set):
    plt.scatter(x[y==i][:, 0], x[y==i][:, 1], marker=m,
                label='class {}'.format(i), edgecolors='k')
plt.show()


