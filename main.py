import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

house_data = pd.read_csv('house_prices.csv')
size = house_data['sqft_living']
price = house_data['price']

x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

regression_model_mse = mean_squared_error(x, y)
print('MSE: ', math.sqrt(regression_model_mse))
print('R squared value: ', model.score(x, y))

print(model.coef_[0])
print(model.intercept_[0])

print('Prediction by the model: ', model.predict([[2000]]))

plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Liner Regression')
plt.xlabel('Size')
plt.ylabel('Price')

