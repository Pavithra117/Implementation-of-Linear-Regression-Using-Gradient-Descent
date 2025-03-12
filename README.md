# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Pavithra K
RegisterNumber:  212224240112
*/
```
```
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    
 X=np.c_[np.ones(len(X1)),X1]

 theta=np.zeros(X.shape[1]).reshape(-1,1) 

 for _ in range(num_iters):

  predictions=(X).dot(theta).reshape(-1,1)

  errors=(predictions-y).reshape(-1,1)

  theta-=learning_rate*(1/len(X1))*X.T.dot(errors)

 return theta

data=pd.read_csv('50_Startups.csv')

X=(data.iloc[1:, :-2].values)

X1=X.astype(float)

scaler=StandardScaler()

y=(data.iloc[1:,-1].values).reshape(-1,1)

X1_Scaled=scaler.fit_transform(X1)

Y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)

new_Scaled=scaler.fit_transform(new_data)


prediction=np.dot(np.append(1, new_Scaled), theta)

prediction=prediction.reshape(-1,1)

pre=scaler.inverse_transform(prediction)

print(f"Predicted value: {pre}")

```

## Output:
![linear regression using gradient descent](sam.png)


![WhatsApp Image 2025-03-12 at 17 52 40_f315d61b](https://github.com/user-attachments/assets/f7b4542e-60a4-4e76-971b-a2ecaa1ca1fd)
![WhatsApp Image 2025-03-12 at 17 52 14_98d7e438](https://github.com/user-attachments/assets/51dbb4f9-4d42-4bdd-bca3-e47c33d4ccf4)
![WhatsApp Image 2025-03-12 at 17 51 42_4ed0ce93](https://github.com/user-attachments/assets/4407601a-c9c4-49c8-a897-d45ec4093af8)


![Screenshot 2025-03-12 180944](https://github.com/user-attachments/assets/b9fb0634-96c2-4236-b9be-1d1961e40f9a)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
