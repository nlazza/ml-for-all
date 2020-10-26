import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sklearn.metrics as metrics


def check_statitionarity(df):
    """Perform a Dickey-Fuller Test to help verifying the stationarity of a TS
    p-value > 0.05 is an indication of non-stationairety

    Parameters
    ----------
    df : DataFrame
        Time series data
    """

    print("Dickey-Fuller Test")
    results = adfuller(df)
    output_df = pd.Series(results[0:4], index=['Test Statistic', 'p-value',
                          'Lags Used', 'Nr. Observations Used'])
    for key, value in results[4].items():
        output_df[f'Critical Value ({key})'] = value
    print(output_df)
    return


def prediction_errors(y_true, y_pred):
    """ Calculate several prediction errors

    Parameters
    ----------
    y_true : array-like containing real values
    y_pred : array-like containing predicted values
    """

    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    r2_score = metrics.r2_score(y_true, y_pred)
    print(f"r2 : {r2_score:.5f}")
    print(f"MAE : {mean_absolute_error:.5f}")
    print(f"RMSE : {rmse:.5f}")
    return

# Read the Spotify Data
df = pd.read_csv("../data/spotify_top200.csv")
# Set the target to predict (one of the numerical columns)
target = "acousticness"
# Drop rows with missing values
df = df[["date", target]].dropna()
# Transform the "date" column to pandas DateTime
df["date"] = df["date"].apply(lambda x: pd.to_datetime(x[:10],
                              format='%Y-%m-%d'))
# Average values from the same week and create a Time Series (TS)
df = df.groupby(by="date").mean().reset_index()
df = df.set_index("date")

# Keep the last 8 weeks of the TS as validation
validation_months = 8
validation_df = df.tail(validation_months)
df.drop(df.tail(validation_months).index, inplace=True)

# Plot the TS
df.plot()
plt.show()

# Check for TS stationarity with "Dickey-Fuller Test"
# p-value < 0.05 is a sign of stationariety
check_statitionarity(df)
# TS is not stationary, p-value of Dickey-Fuller Test >> 0.05
# Use differencing to identify the d term (I) of ARIMA
# Try 1st and 2nd order differencing
fig, axes = plt.subplots(3, 1)
fig.tight_layout()
df_diff = df.diff()
df_diff_diff = df_diff.diff()
df.plot(ax=axes[0])
df_diff.plot(ax=axes[1])
df_diff_diff.plot(ax=axes[2])
axes[0].set_title('Original Series')
axes[1].set_title('1st order differencing')
axes[2].set_title('2nd order differencing')
plt.show()
# After 1 order differentiating, the TS becomes stationary
# d terms for ARIMA = 1

# Checking for AR and MA terms
# Use ACF and PACF plots of stationary (i.e. after differentiating) TS
fig, axes = plt.subplots(2, 1)
plot_acf(df_diff.dropna(), ax=axes[0])
plot_pacf(df_diff.dropna(), ax=axes[1])
axes[0].set_title('ACF')
axes[1].set_title('PACF')
plt.show()
# ACF at lag 1 is negative, suggesting that the series is "overdifferenced"
# MA model without AR (p = 0)
# PACF drop after lag 1 -> q = 1

# Selected model:
# ARIMA(0,1,1)
p = 0
q = 1
d = 1
model = ARIMA(df[target], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())
# AR p-values are < 0.05 indicating a good model

# Plot the residuals to ensure there are no patterns
# (i.e. constant mean and variance)
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
# The residual errors seem fine with near zero mean and uniform variance.

# Plot Actual vs. Fitted data to check how well the ARIMA fitted
# the training data
fitted = model_fit.predict()
actual = df[target]
fig, axes = plt.subplots(1, 1)
fitted.plot(ax=axes, label=f"Fitted {target}")
actual.plot(ax=axes, label=f"Actual {target}")
axes.set_title('Actual vs. Fitted data')
plt.legend(loc=0)
plt.show()

# Predict the future using rolling forecast
# After predicting one step ahead, re-train using the actual target value
predicted_target = []
for i in range(len(validation_df)):
    model = ARIMA(df[target], order=(p, d, q))
    model_fit = model.fit()
    predicted_target.append(model_fit.forecast().values[0])
    df = df.append(validation_df.head(i))
pred_df = pd.Series(predicted_target, index=validation_df.index, name=target)

# Plot Actual vs. Predicted data to check how well the ARIMA predicted
# the validation data
fig, axes = plt.subplots(1, 1)
pred_df.plot(ax=axes, label=f"Predicted {target}")
validation_df.plot(ax=axes, label=f"Actual {target}")
plt.legend(loc=0)
plt.show()

# Calculate the prediction errors
prediction_errors(validation_df[target], predicted_target)
