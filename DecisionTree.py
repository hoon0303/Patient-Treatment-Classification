import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # 결정 트리
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

patient = pd.read_csv('training_set.csv') # csv 파일 읽기

print(patient) # 데이터 출력
print(patient.head()) # 데이터 헤드 출력
print(patient.info()) # 데이터 타입 정보 출력

SEX = {"M" : 1, "F" : 0} # 남성 = 1, 여성 = 0
patient["SEX"] = patient["SEX"].map(SEX) # SEX 전처리

print(patient.describe()) # 데이터 정보 출력

data = patient[['HAEMATOCRIT','HAEMOGLOBINS','ERYTHROCYTE','LEUCOCYTE','THROMBOCYTE','MCH','MCHC','MCV','AGE','SEX']].to_numpy() # 데이터 numpy 배열로 저장
target = patient['SOURCE'].to_numpy() # 타겟 numpy 배열로 저장

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42, stratify=target) # 훈련 데이터와 테스트 데이터를 나눈다.

print(train_input.shape, test_input.shape) # 훈련 세트와 테스트 세트 크기 확인

ss = StandardScaler()
train_scaled = ss.fit_transform(train_input) # 훈련 세트 전처리
test_scaled = ss.transform(test_input) # 테스트 세트 전처리

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target) # 결정 트리 훈련
print(dt.score(train_scaled, train_target)) # 훈련 세트 점수 결과
print(dt.score(test_scaled, test_target)) # 테스트 세트 점수 결과

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['HAEMATOCRIT','HAEMOGLOBINS','ERYTHROCYTE','LEUCOCYTE','THROMBOCYTE','MCH','MCHC','MCV','AGE'])
plt.show()

dt_depth4 = DecisionTreeClassifier(max_depth=3, random_state=42) # 깊이를 3으로 설정
dt_depth4.fit(train_scaled, train_target) # 결정 트리 훈련
print(dt_depth4.score(train_scaled, train_target)) # 훈련 세트 점수 결과
print(dt_depth4.score(test_scaled, test_target)) # 테스트 세트 점수 결과

plt.figure(figsize=(20,15))
plot_tree(dt_depth4, filled=True, feature_names=['HAEMATOCRIT','HAEMOGLOBINS','ERYTHROCYTE','LEUCOCYTE','THROMBOCYTE','MCH','MCHC','MCV','AGE'])
plt.show()