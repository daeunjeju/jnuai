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


def drop(df, column):  # 불필요한 컬럼 삭제하기
    df.drop(column, axis=1, inplace=True)
    # axis=1 : 컬럼을 의미. inplace=True : 삭제한 후 데이터 프레임에 반영
    print(df.head(5))


def heatmap(df):
    print(df.corr())  # 데이터 간의 상관관계(correlation)를 나타냄

    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')
    plt.show()  # 상관관계 함수의 출력을 시각화하는 seaborn 라이브러리의 기능. 색이 옅어지고 값이 1.0에 가까워질수록 데이터 간의 관계가 증가


def hist(a):
    a.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    plt.show()


def violinplot(df, a, b):  # violinplot 그래프
    plt.figure(figsize=(5, 4))
    plt.subplot(1, 1, 1)
    sns.violinplot(x=a, y=b, data=df)
    plt.show()


def scatter(df, column_x, column_y):  # scatter 그래프 그리기
    df.plot.scatter(x=column_x, y=column_y, figsize=(10, 5))
    plt.xticks(np.arange(0, 2, 1))
    plt.show()


def boxplot( df, column, column_by):  # boxplot 그래프 그리기
    df.boxplot(column=column, by=column_by, figsize=(10, 5))
    plt.xticks(np.arange(0, 2, 1))
    plt.show()


def crosstab(df, column_x, column_y):  # crosstab 그래프 그리기
    pd.crosstab(df[column_x], df[column_y], margins=True)
    plt.show()


def pairplot(df):  # pairplot 그래프 그리기
    sns.pairplot(df)
    plt.show()
    plt.xticks(np.arange(0, 2, 1))
    plt.show()


# 전체데이터를 학습용(70%), 테스트 용(30%)으로 나눔
def split(df):
    a, b = train_test_split(df, test_size=0.3)
    # train=70% and test=30%
    return a, b


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
