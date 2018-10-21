# Hubble Trouble Multilinregress

import pandas as pd
import numpy as np
import sklearn as sk
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn import metrics

star = pd.read_csv("stars_data.csv")
star = star.dropna()

# describes the statisccs of the dataset
star.describe()

# independant variables
M = star.iloc[:, 0]

D = star.iloc[:, 2]
D = np.multiply(D, 0.306601)

# dependant variable
m = star.iloc[:, 1]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  
## Linear regression
#regressor = sk.linear_model.LinearRegression()
#
#regressor.fit(X_train, Y_train)
#
## Coeeficients for each of the varibales
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
#
#print(coeff_df)
#
#Y_pred = regressor.predict(X_test)
D[D==0] = 1
m_pred = 5*np.log10(D)-5+M

df = pd.DataFrame({'Actual': m, 'Predicted': m_pred})  
print(df)  
  
print('Mean Absolute Error:', metrics.mean_absolute_error(m, m_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(m, m_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(m, m_pred)))

plt.figure(1)
plt.title("Projected Model")
plt.plot(df.index, df.iloc[:, 0], 'bo', label="Expected")
plt.plot(df.index, df.iloc[:, 1], 'go', label="Predicted")
plt.xlabel("Star Index Number")
plt.ylabel("Apparent Magnitude")
plt.legend()
plt.show()

