import numpy as np
import pandas as pdpip 
import pyomo.environ as pyo
import romodel

filename = r"C:\Users\garan\Desktop\Ponts\data_test.csv"

df = pd.read_csv(filename, sep=';')
df_new = pd.DataFrame(columns=['time', "P10", "P90", 'Most recent forecast'])

list = []
for date in df["Datetime"]:
    list.append(date[11:16])

df_new['time'] = list
df_new['P10'] = df["Most recent P10"]
df_new['P90'] = df["Most recent P90"]
df_new['Most recent forecast'] = df['Most recent forecast']

print(len(df_new))
unc_set = []
for row in range(24):
    mean_P10 = 0
    mean_P90 = 0
    for j in range(4):
        mean_P10 += df_new['P10'][4*row+j]
        mean_P90 += df_new['P90'][4*row+j]
    unc_set.insert(0,  [mean_P10/4, mean_P90/4])

print(unc_set[:15])

most_recent_forecast = np.array(df_new['Most recent forecast'])

model = pyo.ConcreteModel()
model.U = UncSet()

model.ppa = UncParam(range(24), nominal = most_recent_forecast, uncset = model.U)

for i in range(24):
    name_sup = 'sup' + str(i)
    name_inf = 'inf' + str(i)
    model.U.name_sup = pyo.Constraint(expr = model.ppa[i] <= bound_set[i][1])
    model.U.name_inf = pyo.Constraint(expr = model.ppa[i] >= bound_set[i][0])