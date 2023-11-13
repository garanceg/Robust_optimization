import datetime as dt
from elia import elia

connection = elia.EliaPandasClient()
start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 1, 15)

df = connection.get_historical_wind_power_estimation_and_forecast(start=start, end=end)
print(type(df))
print(df.head())