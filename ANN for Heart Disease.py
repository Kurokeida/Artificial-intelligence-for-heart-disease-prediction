import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"")

unique_values_for_chest_pain = []
for y in data["cp"]:
    if y not in unique_values_for_chest_pain:
        unique_values_for_chest_pain.append(y)

le = LabelEncoder()
sc = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore')
ann = tf.keras.models.Sequential()

age_group = []
for x in data["age"]:
    if x > 0 and x < 11:
        age_group.append(0)
    if x > 11 and x < 21:
        age_group.append(1)
    if x > 21 and x < 31:
        age_group.append(3)
    if x > 31 and x < 41:
        age_group.append(4)
    if x > 41 and x < 51:
        age_group.append(5)
    if x > 51 and x < 61:
        age_group.append(6)
    if x > 61 and x < 71:
        age_group.append(7)
    if x > 71 and x < 81:
        age_group.append(8)
    if x > 81 and x < 91:
        age_group.append(9)
    if x > 91:
        age_group.append(10)
        
    if x == 11:
        age_group.append(0)
    if x == 21:
        age_group.append(1)
    if x == 31:
        age_group.append(3)
    if x == 41:
        age_group.append(4)
    if x == 51:
        age_group.append(5)
    if x == 61:
        age_group.append(6)
    if x == 71:
        age_group.append(7)
    if x == 81:
        age_group.append(8)
    if x == 91:
        age_group.append(9)

print(len(data["age"]))
print(len(age_group))
print(max(data["age"]))
data["age"] = age_group

#resting blood pressure
print(min(data["trestbps"]))
print(max(data["trestbps"]))

resting_blood_pressure_brackets = []

for a in data["trestbps"]:
    if a < 100:
        resting_blood_pressure_brackets.append(0)
    if a > 100 and a < 110:
        resting_blood_pressure_brackets.append(1)
    if a > 110 and a< 120:
        resting_blood_pressure_brackets.append(2)
    if a > 120 and a< 130:
        resting_blood_pressure_brackets.append(3)
    if a > 130 and a< 140:
        resting_blood_pressure_brackets.append(4)
    if a > 140 and a< 150:
        resting_blood_pressure_brackets.append(5)
    if a > 150 and a< 160:
        resting_blood_pressure_brackets.append(6)
    if a > 160 and a< 170:
        resting_blood_pressure_brackets.append(7)
    if a > 170 and a< 180:
        resting_blood_pressure_brackets.append(8)
    if a > 180 and a<190:
        resting_blood_pressure_brackets.append(9)
    if a > 190 and a<200:
        resting_blood_pressure_brackets.append(10)
    if a >200:
        resting_blood_pressure_brackets.append(11)
        
    if a == 100:
        resting_blood_pressure_brackets.append(0)
    if a == 110:
        resting_blood_pressure_brackets.append(1)
    if a == 120:
        resting_blood_pressure_brackets.append(2)
    if a == 130:
        resting_blood_pressure_brackets.append(3)
    if a == 140:
        resting_blood_pressure_brackets.append(4)
    if a == 150:
        resting_blood_pressure_brackets.append(5)
    if a == 160:
        resting_blood_pressure_brackets.append(6)
    if a == 170:
        resting_blood_pressure_brackets.append(7)
    if a == 180:
        resting_blood_pressure_brackets.append(8)
    if a == 190:
        resting_blood_pressure_brackets.append(9)
    if a == 200:
        resting_blood_pressure_brackets.append(10)

resting_blood_pressure_brackets_unique = []
print(resting_blood_pressure_brackets)

for i in resting_blood_pressure_brackets:
    if i not in resting_blood_pressure_brackets_unique:
        resting_blood_pressure_brackets_unique.append(i)

print(resting_blood_pressure_brackets_unique)
print(len(data["trestbps"]))
print(len(resting_blood_pressure_brackets))
data["trestbps"] = resting_blood_pressure_brackets

print(min(data["chol"]))
print(max(data["chol"]))

chol_brackets = []

for x1 in data["chol"]:
    if x1 < 100:
        chol_brackets.append(0)
    if x1 > 100 and x1 < 150:
        chol_brackets.append(0)
    if x1 > 150 and x1 < 200:
        chol_brackets.append(1)
    if x1 > 200 and x1 < 250:
        chol_brackets.append(2)
    if x1 > 250 and x1 < 300:
        chol_brackets.append(3)
    if x1 > 300 and x1 < 350:
        chol_brackets.append(4)
    if x1 > 350 and x1 < 400:
        chol_brackets.append(5)
    if x1 >400 and x1 < 450:
        chol_brackets.append(6)
    if x1 > 450 and x1 < 500:
        chol_brackets.append(7)
    if x1 >500 and x1< 550:
        chol_brackets.append(8)
    if x1 >550:
        chol_brackets.append(9)

    if x1 == 100:
        chol_brackets.append(0)
    if x1 == 150:
        chol_brackets.append(1)
    if x1 == 200:
        chol_brackets.append(2)
    if x1 == 250:
        chol_brackets.append(3)
    if x1 == 300:
        chol_brackets.append(4)
    if x1 == 350:
        chol_brackets.append(5)
    if x1 == 400:
        chol_brackets.append(6)
    if x1 == 450:
        chol_brackets.append(7)
    if x1 == 500:
        chol_brackets.append(8)
    if x1 == 550:
        chol_brackets.append(9)

chol_brackets_unique = []
for x2 in chol_brackets:
    if x2 not in chol_brackets_unique:
        chol_brackets_unique.append(x2)
        
print(len(data["chol"]))
print(len(chol_brackets))
print(chol_brackets)
print(chol_brackets_unique)
data["chol"] = chol_brackets

print(min(data["thalach"]))
print(max(data["thalach"]))

maximum_heart_rate_achieved_bracket = []

for x3 in data["thalach"]:
    if x3 < 100:
        maximum_heart_rate_achieved_bracket.append(0)
    if x3 > 100 and x3 < 150:
        maximum_heart_rate_achieved_bracket.append(1)
    if x3 > 150 and x3 < 200:
        maximum_heart_rate_achieved_bracket.append(2)
    if x3 > 200:
        maximum_heart_rate_achieved_bracket.append(3)
        
    if x3 == 100:
        maximum_heart_rate_achieved_bracket.append(0)
    if x3 == 150:
        maximum_heart_rate_achieved_bracket.append(1)
    if x3 == 200:
        maximum_heart_rate_achieved_bracket.append(2)
    
print(len(maximum_heart_rate_achieved_bracket))
data["thalach"] = maximum_heart_rate_achieved_bracket

print(max(data["oldpeak"]))
print(data.iloc[:, 0:-1])

print(data.iloc[:, -1])

training_set = data.iloc[:, 0:-1]
testing_set = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(training_set,testing_set, test_size=0.2, random_state=0)

y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train,y_train, batch_size=32, epochs=100, validation_data=(X_test,y_test))

ann.save("HeartDiseaseModel.h5")
ann.load_weights("HeartDiseaseModel.h5")

metrics = pd.DataFrame(ann.history.history)
metrics[['loss','val_loss']].plot()
metrics[["accuracy","val_accuracy"]].plot()

trainingInfo = ann.evaluate(X_train,y_train)

print("Training data")
print("Loss:",trainingInfo[0])
print("Accuracy:",trainingInfo[1])

testingInfo = ann.evaluate(X_test,y_test)

print("testing data")
print("Loss:",testingInfo[0])
print("Accuracy:",testingInfo[1])

predictor = data.iloc[165,0:-1]
prediction= ann.predict(sc.fit_transform([predictor]))

print(prediction)
print(prediction > 0.5)