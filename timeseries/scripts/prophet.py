import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


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
# Prophet requires two columns ['ds', 'y']:
# ds contains date times
# y contains observations
target = "y"
df.columns = ['ds', target]

# Keep the last 8 weeks of the TS as validation
validation_months = 8
validation_df = df.tail(validation_months)
df.drop(df.tail(validation_months).index, inplace=True)

# Plot the TS
df.set_index("ds").plot()
plt.title("Original time series data")
plt.show()

# Fit the Prophet model
model = Prophet()
model.fit(df)

# Plot the fitted model
# Prophet shows the original data as black dots and the blue line is the
# fitted model. The light blue line represents the confidence intervals
forecast = model.predict(df)
fig = model.plot(forecast, uncertainty=True)
# Add_changepoints_to_plot adds vertical lines that are changepoints,
# i.e Prophet identified where the trend changed
a = add_changepoints_to_plot(fig.gca(), model, forecast)
plt.title("Change points of the timeseries")
plt.show()

# Plot the trend and the seasonal component of the forecast
model.plot_components(forecast)
plt.title("Components of the timeseries")
plt.show()

# Ask Prophet to predict future values
# Append 8 "future" dates to the original dataframe
future = pd.concat([df, validation_df])
forecast = model.predict(future)
fig = model.plot(forecast)
plt.title("Future prediction")
plt.show()

# Plot Real vs. Predicted data to check how well Prophet predicted
# the validation data
fig, axes = plt.subplots(1, 1)
pred_df = forecast[["ds", "yhat"]].tail(len(validation_df))
pred_df.set_index('ds').plot(ax=axes)
validation_df.set_index('ds').plot(ax=axes)
h, l = axes.get_legend_handles_labels()
plt.legend(h, ["Predicted target", "Real target"], loc=0)
axes.set_title('Real vs. Predicted data')
plt.show()

# Calculate the prediction errors
prediction_errors(validation_df[target], pred_df['yhat'])
