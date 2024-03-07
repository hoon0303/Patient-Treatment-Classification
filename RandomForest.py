import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
rf.fit(train_input, train_target) # 랜덤 포레스트 훈련
print(rf.score(train_input, train_target)) # 훈련 세트 점수 결과
print(rf.score(test_input, test_target)) # 테스트 세트 점수 결과