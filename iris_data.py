#붓꽃 데이터셋 (scikit-leran)

#데이터적재 
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#load_iris가 반환한 iris 객체는 파이썬의 dictionary와 유사한 bunch 클래스의 객체입니다. 즉 키와 값으로 구성되어 있음
print("iris_datset의 키: \n{}".format(iris_dataset.keys()))
#output - iris_datset의 키:
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

#DESCR의 키에는 데이터셋에 대한 간략한 설명이 들어가 있습니다. 
#앞부분을 간단하게 보려면, 
print(iris_dataset['DESCR'][:193] + "\n...")

#target_names의 값은 우리가 예측하려는 붓꽃 품종의 이름을 문자열 배열로 가지고 있습니다. 
print("타깃의 이름: {}".format(iris_dataset['target_names']))

#feature_names는 각 특성을 설명하는 문자열 리스트 입니다. 
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))

#실제 데이터는 target과 data 필드에 들어가 있습니다. 
#data는 꽃잎의 길이와 폭, 꽃받침의 길이와 폭을 수치 값으로 가지고 있는 numpy 배열입니다. 
print("data 타입: {}".format(type(iris_dataset['data']))) #numpy.ndarray

#data 배열의 행은 개개의 꽃이 되며 열은 각 꽃에서 구한 네개의 측정치 입니다. 
print("data의 크기: {}".format(iris_dataset['data'].shape)) #(150,4)

#150개의 봋꽃 데이터를 가지고 있습니다. 
#처음 다섯 샘플의 특성은 다음과 같습니다. 
print("data 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))

#target 배열도 샘플 붓꽃의 품종을 담은 numpy 배열입니다. 
print("target의 타입: {}".format(type(iris_dataset['target']))) #numpy.ndarray

#붓꽃의 종류는 0에서 2까지의 정수로 기록되어 있습니다. 
#숫자의 의미는 ['target_names'] 배열에서 확인할 수 있습니다. 
#0은 setosa, 1은 versicolor, 2는 virginica 입니다. 
