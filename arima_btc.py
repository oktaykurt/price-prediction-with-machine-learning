# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

data = pd.read_csv('data-sets/BTC-USD-5Y.csv',index_col='Date')
data = data[['Adj Close']]

data.index = pd.date_range(start='2017-11-20', end='2022-11-20')

print(data)

result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')


# %%
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# %%
# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(data.values)
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(data.diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(data.diff().diff())
ax3.set_title('2nd Order Differencing')

# %%
plot_pacf(data.diff().dropna())


# %%
plot_acf(data.diff().dropna())

# %%
model = ARIMA(data['2019-11-20':], order=(1, 1, 1))
model_fit = model.fit()

predicted_data = model_fit.predict(start="2020-11-20")
new_data = data['2020-11-20':]
new_data = new_data['Adj Close']

# %%
error = abs(np.divide(
    (np.subtract(new_data.values, predicted_data.values)), new_data.values) * 100)
error_index = new_data.index

# %%
error_df = pd.DataFrame(error, error_index)
plt.plot(error_df, )
plt.title('Error (%)')

fig, ax = plt.subplots()

ax = data.loc['2019-11-20':].plot(ax=ax, color='c')
predicted_data.plot(ax=ax, color='m', title='Prediction')
plt.title('Actual Data vs Prediction')

plt.legend(('Actual Data', 'Prediction'),
           loc='upper right')

plt.show()
