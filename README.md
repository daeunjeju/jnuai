11월의 날씨 정보를 이용하여 입동 여부 알아보기
================

# jnuai
제주대학교 인공지능 기말 과제물입니다.

https://github.com/yungbyun/mllib 의 Classification_util.py 중 제게 필요한 코드를 골라 저만의 myimport.py를 만들어 사용했습니다.

이제 11월의 날씨 정보 중 평균기온, 최저기온, 최고기온, 평균 풍속, 평균 상대습도, 평균 지면온도를 이용하여 입동인지, 입동 전후인지를 알아보려고 합니다.
데이터들은 dataset.csv에서 확인할 수 있습니다.

먼저 필요한 모듈을 불러옵니다

myimport.py :
~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

def ignore_warning():
    import warnings
    warnings.filterwarnings('ignore')

# matplotlib은 한국어 폰트를 지원하지 않기 때문에 matpoltlib을 한국어 지원 폰트로 수정하기
import matplotlib as mpl
import matplotlib.font_manager as fm

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 15
plt.rcParams["figure.figsize"] = (14, 4)
~~~
classification.py :
~~~python
import myimport as my
my.ignore_warning()
~~~
이제 데이터셋을 읽고 분석해봅니다.

myimport.py:
~~~python
def read(filename):  # 데이터셋 읽기
    object = pd.read_csv(filename)
    return object

def info(df):
    print(df.head(5))  # 위에서부터 원하는 수만큼 데이터를 보여줌
    print("------------------------------------")
    df.info()  # 컬럼에 대한 지정된 정보 제공
    print("------------------------------------")
    print(df.columns)  # 컬럼에 대한 정보를 확인할 수 있다
    print("------------------------------------")
~~~
classification.py:
~~~python
df = my.read("dataset.csv") # 데이터셋 읽기
my.info(df)
~~~
![1](https://user-images.githubusercontent.com/47231570/71112546-13c8c500-220f-11ea-9050-bf8284291c47.PNG)

우선 '지점'이라는 컬럼은 필요없는 컬럼이므로 삭제합니다.

myimport.py:
~~~python
def drop(df, column):  # 불필요한 컬럼 삭제하기
    df.drop(column, axis=1, inplace=True)
    # axis=1 : 컬럼을 의미. inplace=True : 삭제한 후 데이터 프레임에 반영
    print(df.head(5))
~~~
classification.py:
~~~python
my.drop(df, '지점') # 분류하는데 불필요한 컬럼 제거하기
my.drop(df, '일시')
~~~
그리고 '입동 여부'는 object이므로 후에 heatmap을 사용하기 위해선 숫자형으로 바꿔야합니다. 나중에 해도 상관없지만 미리 했습니다.

classification.py:
~~~python
df['입동여부'] = df['입동여부'].map({'전':0, '입동':1, '후':2})
print(df.head(5))
df.info()
~~~
![2](https://user-images.githubusercontent.com/47231570/71112840-be40e800-220f-11ea-8294-37975293ae19.PNG)

describe()를 사용하여 컬럼별 총 데이터 수, 평균값, 표준편차, 최소~최대 및 4분위수를 확인해보겠습니다.
classification.py:
~~~python
print(df.describe())
~~~
![3](https://user-images.githubusercontent.com/47231570/71112937-ef211d00-220f-11ea-82f7-af0cd9a0a0cb.PNG)

이제 상관계수를 알아보기 위해 heatmap을 사용하겠습니다.
myimport.py:
~~~python
def heatmap(df):
    print(df.corr())  # 데이터 간의 상관관계(correlation)를 나타냄

    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')
    plt.show()  # 상관관계 함수의 출력을 시각화하는 seaborn 라이브러리의 기능.
                # 색이 옅어지고 값이 1.0에 가까워질수록 데이터 간의 관계가 증가
~~~
classification.py:
~~~python
my.heatmap(df)
~~~
![heatmap](https://user-images.githubusercontent.com/47231570/71113062-28598d00-2210-11ea-8a37-c7b29d5aa483.png)

heatmap을 이용하여 입동 여부와 다른 컬럼들 중 '평균 풍속'과 '평균 상대습도'를 제외한 컬럼들은 낮은 상관계수를 보였습니다.
이를 확인하기 위해 그래프를 이용하여 비교하겠습니다.

먼저 numpy의 histogram을 이용한 것입니다. 각 컬럼들의 histogram을 볼 수 있습니다.
myimport.py:
~~~python
def hist(a):
    a.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    plt.show()
~~~
classification.py:
~~~python
my.hist(df)
~~~
![hist](https://user-images.githubusercontent.com/47231570/71113162-6060d000-2210-11ea-84c1-84046ad0b594.png)

다음으로 입동여부와 평균기온을 scatter plot을 이용해 그려보았습니다.
myimport.py:
~~~python
def scatter(df, column_x, column_y):  # scatter 그래프 그리기
    df.plot.scatter(x=column_x, y=column_y, figsize=(10, 5))
    plt.xticks(np.arange(0, 2, 1))
    plt.show()
~~~
classification.py:
~~~python
my.scatter(df, '입동여부', '평균기온')
~~~
![평균기온vs입동여부](https://user-images.githubusercontent.com/47231570/71113410-dbc28180-2210-11ea-8149-96fd891db897.png)

다음은 violin plot을 이용한 입동여부와 최저기온입니다.
myimport.py:
~~~python
def violinplot(df, a, b):  # violinplot 그래프
    plt.figure(figsize=(5, 4))
    plt.subplot(1, 1, 1)
    sns.violinplot(x=a, y=b, data=df)
    plt.show()
~~~
classification.py:
~~~python
my.violinplot(df, '입동여부', '최저기온')
~~~
![최저기온vs입동여부](https://user-images.githubusercontent.com/47231570/71113486-06acd580-2211-11ea-8e66-2802f162f41a.png)

입동여부와 최고기온 그래프입니다.
classification.py:
~~~python
my.violinplot(df, '입동여부', '최고기온')
~~~
![최고기온vs입동여부](https://user-images.githubusercontent.com/47231570/71113639-61dec800-2211-11ea-89fe-ccea998d3db6.png)
입동여부와 평균 풍속 그래프입니다.

classification.py:
~~~python
my.scatter(df, '입동여부', '평균 풍속')
~~~
![평균풍속vs입동여부](https://user-images.githubusercontent.com/47231570/71113655-686d3f80-2211-11ea-8afe-71f274acc941.png)
입동여부와 평균 상대습도 그래프입니다.

classification.py:
~~~python
my.scatter(df, '입동여부', '평균 상대습도')
~~~
![평균상대습도vs입동여부](https://user-images.githubusercontent.com/47231570/71113664-6b683000-2211-11ea-8893-89ae3f1e505a.png)

입동여부와 평균 지면온도 그래프입니다.
classification.py:
~~~python
my.violinplot(df, '입동여부', '평균 지면온도')
~~~
![평균지면온도vs입동여부](https://user-images.githubusercontent.com/47231570/71113666-6b683000-2211-11ea-8b30-45306828ef3e.png)

pairplot을 이용한 각 컬럼들의 그래프입니다.
myimport.py:
~~~python
def pairplot(df):  # pairplot 그래프 그리기
    sns.pairplot(df)
    plt.show()
    plt.xticks(np.arange(0, 2, 1))
    plt.show()
~~~
classification.py:
~~~python
my.pairplot(df)
~~~
![pairplot](https://user-images.githubusercontent.com/47231570/71113804-b5511600-2211-11ea-8db1-3422c753d806.png)

데이터를 확인했으니 이제 분류를 하기 위해서 데이터프레임을 분리해야 합니다.
myimport.py:
~~~python
def split(df):
    a, b = train_test_split(df, test_size=0.3)
    # train=70% and test=30%
    return a, b
~~~
classification.py:
~~~python
train, test = my.split(df)
print(train.head(5))
print(test.head(5))
~~~



myimport.py:
~~~python

~~~
classification.py:
~~~python

~~~







my.ignore_warning() # 분류를 위한 알고리즘을 사용할 때 나타나는 경고 메세지를 필터링하기 위해 사용

my.run_logistic_regression(df, ['평균 풍속', '평균 상대습도'], '입동여부')
my.run_decision_tree_classifier(df, ['평균 풍속', '평균 상대습도'], '입동여부')
my.run_neighbor_classifier(df, ['평균 풍속', '평균 상대습도'], '입동여부', 5)
my.run_svm(df, ['평균 풍속', '평균 상대습도'], '입동여부')

~~~python
def run_logistic_regression(df, list, target):
    train, test = train_test_split(df, test_size=0.3)
    train_X = train[list]  # 키와 발크기만 선택
    train_y = train[target]  # 정답 선택
    test_X = test[list]  # taking test data features
    test_y = test[target]  # output value of test data
    baby1 = LogisticRegression()  # 애기
    baby1.fit(train_X, train_y)  # 가르친 후
    prediction = baby1.predict(test_X)  # 테스트
    rate = metrics.accuracy_score(prediction, test_y) * 100
    print('인식률:', rate)
    return rate

def run_neighbor_classifier(df, list, target, num):
    train, test = train_test_split(df, test_size=0.3)
    train_X = train[list]  # 키와 발크기만 선택
    train_y = train[target]  # 정답 선택
    test_X = test[list]  # taking test data features
    test_y = test[target]  # output value of test data
    baby1 = KNeighborsClassifier(n_neighbors=num)  # 애기
    baby1.fit(train_X, train_y)  # 가르친 후
    prediction = baby1.predict(test_X)  # 테스트
    rate = metrics.accuracy_score(prediction, test_y) * 100
    print('인식률:', rate)
    return rate

def run_decision_tree_classifier(df, list, target):
    train, test = train_test_split(df, test_size=0.3)
    train_X = train[list]  # 키와 발크기만 선택
    train_y = train[target]  # 정답 선택
    test_X = test[list]  # taking test data features
    test_y = test[target]  # output value of test data
    baby1 = DecisionTreeClassifier()  # 애기
    baby1.fit(train_X, train_y)  # 가르친 후
    prediction = baby1.predict(test_X)  # 테스트
    rate = metrics.accuracy_score(prediction, test_y) * 100
    print('인식률:', rate)
    return rate

def run_svm(df, list, target):
    train, test = train_test_split(df, test_size=0.3)
    train_X = train[list]  # 키와 발크기만 선택
    train_y = train[target]  # 정답 선택
    test_X = test[list]  # taking test data features
    test_y = test[target]  # output value of test data
    baby1 = svm.SVC()  # 애기
    baby1.fit(train_X, train_y)  # 가르친 후
    prediction = baby1.predict(test_X)  # 테스트
    rate = metrics.accuracy_score(prediction, test_y) * 100
    print('인식률:', rate)
    return rate
~~~
