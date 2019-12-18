import myimport as my

my.ignore_warning()

df = my.read("dataset.csv") # 데이터셋 읽기

my.info(df) # 데이터셋에 대해 분석하기

# 다음은 컬럼들의 단위입니다.
# 평균기온 : °C
# 최저기온 : °C
# 최고기온 : °C
# 평균 풍속 : m/s
# 평균 상대습도 : %
# 평균 지면온도 : °C

my.drop(df, '지점') # 분류하는데 불필요한 컬럼 제거하기
my.drop(df, '일시')

# 입동 여부의 전, 입동, 후이 object이므로 숫자형으로 바꾸어줘야 합니다.
df['입동여부'] = df['입동여부'].map({'전':0, '입동':1, '후':2})
print(df.head(5))
df.info()

print(df.describe()) #(수로 계산이 가능한) 각 칼럼별로 총 데이터 수, 평균값, 표준편차, 최소~최대 및 4분위수가 나온다.

my.heatmap(df)
# heatmap을 이용하여 입동 여부와 다른 컬럼들 중 평균 풍속과 평균 상대습도를 제외한 컬럼들은 낮은 상관계수를 보였습니다.
# 이를 확인하기 위해 그래프를 이용하여 비교해 볼 것입니다.

my.hist(df)
my.scatter(df, '입동여부', '평균기온') # scatter를 이용한 입동여부와 평균기온 그래프
my.violinplot(df, '입동여부', '최저기온') # violinplot을 이용한 입동여부와 최저기온 그래프
my.violinplot(df, '입동여부', '최고기온') # violinplot을 이용한 입동여부와 최고기온 그래프
my.scatter(df, '입동여부', '평균 풍속')

my.scatter(df, '입동여부', '평균 상대습도') # scatter를 이용한 입동여부와 평균 상대습도 그래프
my.violinplot(df, '입동여부', '평균 지면온도') # violinplot을 이용한 입동여부와 평균 지면온도 그래프

my.pairplot(df)

train, test = my.split(df) # 데이터 문제지와 정답지로 분리하기
print(train.head(5))
print(test.head(5))

my.ignore_warning() # 분류를 위한 알고리즘을 사용할 때 나타나는 경고 메세지를 필터링하기 위해 사용

my.run_logistic_regression(df, ['평균 풍속', '평균 상대습도'], '입동여부')
my.run_decision_tree_classifier(df, ['평균 풍속', '평균 상대습도'], '입동여부')
my.run_neighbor_classifier(df, ['평균 풍속', '평균 상대습도'], '입동여부', 5)
my.run_svm(df, ['평균 풍속', '평균 상대습도'], '입동여부')
