import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
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
        output_df[f"Critical Value ({key})"] = value
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

# Read the Champagne Sales Data Time Series (TS)
df = pd.read_csv("data/champagne_sales.csv", sep="\t")
# Transform the "Month" column to pandas DateTime and set it as index
df["Month"] = df["Month"].apply(lambda x: pd.to_datetime(x, format='%y-%m'))
df = df.set_index("Month")

# Keep the last 6 months of the TS as validation
validation_months = 6
validation_df = df.tail(validation_months)
df.drop(df.tail(validation_months).index, inplace=True)

# Plot the TS
df.plot()
plt.show()

# Check for TS stationarity with "Dickey-Fuller Test"
check_statitionarity(df)

# TS is not stationary, evidence of seasonality at 12 months
# Use differencing to identify the d term (I) of ARIMA
# Try 1st order seasonal differencing and its first order differencing as well
fig, axes = plt.subplots(3, 1)
fig.tight_layout()
saesonal_diff = df.diff(periods=12)
saesonal_diff_diff = saesonal_diff.diff()
df.plot(ax=axes[0])
saesonal_diff.plot(ax=axes[1])
saesonal_diff_diff.plot(ax=axes[2])
axes[0].set_title('Original Series')
axes[1].set_title('Seasonal 1st order differencing')
axes[2].set_title('Non-Seasonal 1st order differencing')
plt.show()

# After differentiating, the TS becomes stationary
# d terms for ARIMA:
# Seasonal d = 1
# Non-seasonal d = 1

# Checking for AR and MA terms
# Use ACF and PACF plots of stationary TS (seasonal part)
fig, axes = plt.subplots(2, 1)
plot_acf(saesonal_diff.dropna(), ax=axes[0])
plot_pacf(saesonal_diff.dropna(), ax=axes[1])
axes[0].set_title('ACF Seasonal')
axes[1].set_title('PACF Seasonal')
plt.show()
# ACF at lag 1 is positive -> using an AR model without MA (q = 0)
# PACF drop after lag 1 -> p = 1

# Use ACF and PACF plots of stationary TS (non-seasonal part)
fig, axes = plt.subplots(2, 1)
plot_acf(saesonal_diff_diff.dropna(), ax=axes[0])
plot_pacf(saesonal_diff_diff.dropna(), ax=axes[1])
axes[0].set_title('ACF Non-Seasonal')
axes[1].set_title('PACF Non-Seasonal')
plt.show()
# ACF at lag 1 is positive -> AR model without MA (q = 0)
# PACF drop after lag 1 -> p = 1

# Selected model:
# SARIMA(1,1,0)(1,1,0)12
model = sm.tsa.statespace.SARIMAX(df["Champagne Sales"],
                                  order=(1, 1, 0),
                                  seasonal_order=(1, 1, 0, 12))
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
actual = df["Champagne Sales"]
fig, axes = plt.subplots(1, 1)
fitted.plot(ax=axes, label="Fitted sales")
actual.plot(ax=axes, label="Actual sales")
axes.set_title('Actual vs. Fitted data')
plt.legend(loc=0)
plt.show()

# Predict the future Champagne Sales for the validation period
pred_validation = model_fit.forecast(len(validation_df))

# Plot Actual vs. Predicted data to check how well the ARIMA predicted
# the validation data
fig, axes = plt.subplots(1, 1)
# Uncomment if you want to plot the whole series + validation
#df["Champagne Sales"].plot(ax=axes, label="Training")
pred_validation.plot(ax=axes, label="Predicted sales")
validation_df["Champagne Sales"].plot(ax=axes, label="Actual sales")
plt.legend(loc=0)
plt.show()

# Calculate the prediction errors
prediction_errors(validation_df["Champagne Sales"], pred_validation)
