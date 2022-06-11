from google.colab import files
data_to_load=files.upload()

import pandas as pd
import csv
import plotly.express as px
import statistics
df=pd.read_csv("escape_velocity-hw.csv")
Velocity=df["Velocity"].tolist()
Escaped=df["Escaped"].tolist()
graph1=px.scatter(x=Velocity,y=Escaped)
graph1.show()

import numpy as np
Velocity_array=np.array(Velocity)
Escaped_array=np.array(Escaped)
m,c=np.polyfit(Velocity_array,Escaped_array,1)
y=[]
for x in Velocity_array :
  y_value=m*x+c
  y.append(y_value)

graph2=px.scatter(x=Velocity_array,y=Escaped_array)
graph2.update_layout(shapes=[dict(type='line',y0=min(y),y1=max(y),x0=min(Velocity_array),x1=max(Velocity_array))])
graph2.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
Velocity=df["Velocity"].tolist()
Escaped=df["Escaped"].tolist()
X=np.reshape(Velocity,(len(Velocity),1))
Y=np.reshape(Escaped,(len(Escaped),1))
lr=LogisticRegression()
lr.fit(X,Y)
plt.figure()
plt.scatter(X.ravel(),Y,color='black',zorder=20)

def Model (x):
  return 1/(1+np.exp(-x))
X_test=np.linspace(0,100,200)
chances=Model(X_test*lr.coef_+lr.intercept_).ravel()
plt.plot(X_test,chances,color='red',linewidth=3)
plt.axhline(y=0,color='k',linestyle='-')
plt.axhline(y=1,color='k',linestyle='-')
plt.axhline(y=0.5,color='b',linestyle='--')
plt.axvline(x=X_test[25],color='b',linestyle='--')
plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0,30)
plt.show()
