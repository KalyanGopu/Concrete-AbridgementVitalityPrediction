import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
path_name = input("Enter the path name:")
print("Entered path name",path_name)
df = pd.read_csv(path_name)
df.head()
df.shape
df.info()
df.isna().sum()
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
for i in df.columns:
    for j in df.columns:
        plt.figure(figsize=(9,7))
        sns.scatterplot(x=i,y=j,hue="concrete_compressive_strength",data=df)
        plt.show()
def outlier(data,column):
    plt.figure(figsize=(5,3))
    sns.boxplot(data[column])
    plt.title("{} distribution".format(column))
for i in df.columns:
    outlier(df,i)
def end_value_show(data,column):
    print("min value of {} is {} \nmax value of {} is {}".format(column,data[column].min(),column,data[column].max()))
for i in df.columns:
    end_value_show(df,i)
df=df[df["blast_furnace_slag"]<350]
df=df[(df["water"]<246) & (df["water"]>122)]
df=df[df["superplasticizer"]<25]
df=df[df["age"]<150]
df.columns
df.drop(["blast_furnace_slag"],axis=1,inplace=True)
df.drop(["coarse_aggregate"],axis=1,inplace=True)
df.drop(["fine_aggregate "],axis=1,inplace=True)
df.columns
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
x=df.drop(["concrete_compressive_strength"],axis=1)
y=df["concrete_compressive_strength"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train.shape
from tensorflow.keras import models,layers
model=models.Sequential()
model.add(layers.Dropout(0.1))
model.add(layers.Dense(100,activation='relu',input_shape=(x_train.iloc[1].shape)))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(5,activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)
pred=model.predict(x_test)
pred[4]
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)
reg.fit(x=x_train, y=y_train, verbose=0)
# evaluate the model
mae, _  = reg.evaluate(x_test, y_test, verbose=0)
#print('MAE: %.3f' % mae)
# use the model to make a prediction
yhat_test = reg.predict(x_test)
# get the best performing model
model = reg.export_model()
# summarize the loaded model
model.summary()
yhat_test
y_test




