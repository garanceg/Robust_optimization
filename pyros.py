import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pyomo.contrib.pyros as pyros

# Importing data and cleaning data
###############################################################################################################################
filename = r"C:\Users\garan\Desktop\Ponts\data_test.csv"

df = pd.read_csv(filename, sep=';')
df_new = pd.DataFrame(columns=['time', "P10", "P90", 'Most recent forecast'])

tlist = []
for date in df["Datetime"]:
    tlist.append(date[11:16])

df_new['time'] = tlist
df_new['P10'] = df["Most recent P10"]
df_new['P90'] = df["Most recent P90"]
df_new['Most recent forecast'] = df['Most recent forecast']

bound_set = []
most_recent_forecast = []
for row in range(24):
    mean_P10 = 0
    mean_P90 = 0
    mean_mrf = 0
    for j in range(4):
        mean_P10 += df_new['P10'][4*row+j]
        mean_P90 += df_new['P90'][4*row+j]
        mean_mrf += df_new['Most recent forecast'][4*row+j]
    bound_set.insert(0,  [mean_P10/4, mean_P90/4])
    most_recent_forecast.insert(0, mean_mrf/4)


mean_mrf_tot = sum(most_recent_forecast)/len(most_recent_forecast)
infbound = [bound_set[i][0] for i in range(24)]
supbound = [bound_set[i][1] for i in range(24)]
###############################################################################################################################

# Setting constants
###############################################################################################################################
boundppa = most_recent_forecast
prix_ppa = 20
pm = np.array([20.79, 17.41, 16.24, 11.9, 9.77, 15.88, 24.88, 29.7, 35.01, 33.95, 29.9, 29.03, 27.07, 26.43, 27.53, 29.05, 31.42, 39.92, 41.3, 41.51, 39.75, 30.13, 30.36, 32.4])
capex_electrolyseur = 1200000*0.0004/100
capex_batterie = 250000*0.0002/100
capex_stock_gaz = 407*0.000137
eff_el = 0.05
demand = mean_mrf_tot
###############################################################################################################################


# Construct model
pyros_solver = pyo.SolverFactory("pyros")
model = pyo.ConcreteModel()

# Define variables

#stockage gaz
model.sg1 = pyo.Var(within=pyo.Reals,bounds=(0, 0))
model.sg2 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg3 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg4 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg5 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg6 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg7 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg8 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg9 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg10 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg11 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg12 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg13 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg14 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg15 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg16 = pyo.Var(within=pyo.Reals,bounds=(0, 150000))
model.sg17 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg18 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg19 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg20 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg21 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg22 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg23 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))
model.sg24 = pyo.Var(within=pyo.Reals,bounds=(0, 15000))

#stockage electricté
model.se1 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se2 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se3 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se4 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se5 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se6 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se7 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se8 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se9 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se10 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se11 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se12 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se13 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se14 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se15 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se16 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se17 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se18 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se19 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se20 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se21 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se22 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se23 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))
model.se24 = pyo.Var(within=pyo.Reals,bounds=(0, 4200))

#capacité électrolyseur
model.ce1 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce2 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce3 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce4 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce5 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce6 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce7 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce8 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce9 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce10 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce11 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce12 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce13 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce14 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce15 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce16 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce17 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce18 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce19 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce20 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce21 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce22 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce23 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))
model.ce24 = pyo.Var(within=pyo.Reals,bounds=(0, 1000))

#energie marché
model.em1 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em2 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em3 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em4 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em5 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em6 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em7 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em8 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em9 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em10 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em11 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em12 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em13 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em14 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em15 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em16 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em17 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em18 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em19 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em20 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em21 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em22 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em23 = pyo.Var(within=pyo.Reals,bounds=(0, None))
model.em24 = pyo.Var(within=pyo.Reals,bounds=(0, None))

# === Define parameters ===
model.set_of_params = pyo.Set(initialize=range(24))
nominal_values = {t:most_recent_forecast[t] for t in range(24)}
model.ppa = pyo.Param(model.set_of_params, initialize=nominal_values, mutable=True)

# === Specify the objective function ===
model.obj = pyo.Objective(expr=pm[0]*model.em1+pm[1]*model.em2+pm[2]*model.em3+pm[3]*model.em4+pm[4]*model.em5+pm[5]*model.em6+pm[6]*model.em7+pm[7]*model.em8+
                      pm[8]*model.em9+pm[9]*model.em10+pm[10]*model.em11+pm[11]*model.em12+pm[12]*model.em13+pm[13]*model.em14+pm[14]*model.em15+pm[15]*model.em16+
                      pm[16]*model.em17+pm[17]*model.em18+pm[18]*model.em19+pm[19]*model.em20+pm[20]*model.em21+pm[21]*model.em22+pm[22]*model.em23+pm[23]*model.em24+
                      prix_ppa*(model.ppa[1]+model.ppa[2]+model.ppa[3]+model.ppa[4]+model.ppa[5]+model.ppa[6]+model.ppa[7]+model.ppa[8]+model.ppa[9]+model.ppa[10]
                                +model.ppa[11]+model.ppa[12]+model.ppa[13]+model.ppa[14]+model.ppa[15]+model.ppa[16]+model.ppa[17]+model.ppa[18]+model.ppa[19]+model.ppa[20]
                                +model.ppa[21]+model.ppa[22]+model.ppa[23]+model.ppa[0]),sense=pyo.minimize)


# Define constraints
model.init = pyo.Constraint(expr=model.se1 == model.ppa[0]+model.em1)
model.cce1 = pyo.Constraint(expr=model.se1+model.ppa[0]+model.em1+model.sg1*eff_el >= demand)
model.cce2 = pyo.Constraint(expr=model.se2+model.ppa[1]+model.em2+model.sg2*eff_el >= demand)
model.cce3 = pyo.Constraint(expr=model.se3+model.ppa[2]+model.em3+model.sg3*eff_el >= demand)
model.cce4 = pyo.Constraint(expr=model.se4+model.ppa[3]+model.em4+model.sg4*eff_el >= demand)
model.cce5 = pyo.Constraint(expr=model.se5+model.ppa[4]+model.em5+model.sg5*eff_el >= demand)
model.cce6 = pyo.Constraint(expr=model.se6+model.ppa[5]+model.em6+model.sg6*eff_el >= demand)
model.cce7 = pyo.Constraint(expr=model.se7+model.ppa[6]+model.em7+model.sg7*eff_el >= demand)
model.cce8 = pyo.Constraint(expr=model.se8+model.ppa[7]+model.em8+model.sg8*eff_el >= demand)
model.cce9 = pyo.Constraint(expr=model.se9+model.ppa[8]+model.em9+model.sg9*eff_el >= demand)
model.cce10 = pyo.Constraint(expr=model.se10+model.ppa[9]+model.em10+model.sg10*eff_el >= demand)
model.cce11 = pyo.Constraint(expr=model.se11+model.ppa[10]+model.em11+model.sg11*eff_el >= demand)
model.cce12 = pyo.Constraint(expr=model.se12+model.ppa[11]+model.em12+model.sg12*eff_el >= demand)
model.cce13 = pyo.Constraint(expr=model.se13+model.ppa[12]+model.em13+model.sg13*eff_el >= demand)
model.cce14 = pyo.Constraint(expr=model.se14+model.ppa[13]+model.em14+model.sg14*eff_el >= demand)
model.cce15 = pyo.Constraint(expr=model.se15+model.ppa[14]+model.em15+model.sg15*eff_el >= demand)
model.cce16 = pyo.Constraint(expr=model.se16+model.ppa[15]+model.em16+model.sg16*eff_el >= demand)
model.cce17 = pyo.Constraint(expr=model.se17+model.ppa[16]+model.em17+model.sg17*eff_el >= demand)
model.cce18 = pyo.Constraint(expr=model.se18+model.ppa[17]+model.em18+model.sg18*eff_el >= demand)
model.cce19 = pyo.Constraint(expr=model.se19+model.ppa[18]+model.em19+model.sg19*eff_el >= demand)
model.cce20 = pyo.Constraint(expr=model.se20+model.ppa[19]+model.em20+model.sg20*eff_el >= demand)
model.cce21 = pyo.Constraint(expr=model.se21+model.ppa[20]+model.em21+model.sg21*eff_el >= demand)
model.cce22 = pyo.Constraint(expr=model.se22+model.ppa[21]+model.em22+model.sg22*eff_el >= demand)
model.cce23 = pyo.Constraint(expr=model.se23+model.ppa[22]+model.em23+model.sg23*eff_el >= demand)
model.cce24 = pyo.Constraint(expr=model.se24+model.ppa[23]+model.em24+model.sg24*eff_el >= demand)

model.cse1 = pyo.Constraint(expr=model.ce2 == model.se1 - model.se2 + model.ppa[1] + model.em2)
model.cse2 = pyo.Constraint(expr=model.ce3 == model.se2 - model.se3 + model.ppa[2] + model.em3)
model.cse3 = pyo.Constraint(expr=model.ce4 == model.se3 - model.se4 + model.ppa[3] + model.em4)
model.cse4 = pyo.Constraint(expr=model.ce5 == model.se4 - model.se5 + model.ppa[4] + model.em5)
model.cse5 = pyo.Constraint(expr=model.ce6 == model.se5 - model.se6 + model.ppa[5] + model.em6)
model.cse6 = pyo.Constraint(expr=model.ce7 == model.se6 - model.se7 + model.ppa[6] + model.em7)
model.cse7 = pyo.Constraint(expr=model.ce8 == model.se7 - model.se8 + model.ppa[7] + model.em8)
model.cse8 = pyo.Constraint(expr=model.ce9 == model.se8 - model.se9 + model.ppa[8] + model.em9)
model.cse9 = pyo.Constraint(expr=model.ce10 == model.se9 - model.se10 + model.ppa[9] + model.em10)
model.cse10 = pyo.Constraint(expr=model.ce11 == model.se10 - model.se11 + model.ppa[10] + model.em11)
model.cse11 = pyo.Constraint(expr=model.ce12 == model.se11 - model.se12 + model.ppa[11] + model.em12)
model.cse12 = pyo.Constraint(expr=model.ce13 == model.se12 - model.se13 + model.ppa[12] + model.em13)
model.cse13 = pyo.Constraint(expr=model.ce14 == model.se13 - model.se14 + model.ppa[13] + model.em14)
model.cse14 = pyo.Constraint(expr=model.ce15 == model.se14 - model.se15 + model.ppa[14] + model.em15)
model.cse15 = pyo.Constraint(expr=model.ce16 == model.se15 - model.se16 + model.ppa[15] + model.em16)
model.cse16 = pyo.Constraint(expr=model.ce17 == model.se16 - model.se17 + model.ppa[16] + model.em17)
model.cse17 = pyo.Constraint(expr=model.ce18 == model.se17 - model.se18 + model.ppa[17] + model.em18)
model.cse18 = pyo.Constraint(expr=model.ce19 == model.se18 - model.se19 + model.ppa[18] + model.em19)
model.cse19 = pyo.Constraint(expr=model.ce20 == model.se19 - model.se20 + model.ppa[19] + model.em20)
model.cse20 = pyo.Constraint(expr=model.ce21 == model.se20 - model.se21 + model.ppa[20] + model.em21)
model.cse21 = pyo.Constraint(expr=model.ce22 == model.se21 - model.se22 + model.ppa[21] + model.em22)
model.cse22 = pyo.Constraint(expr=model.ce23 == model.se22 - model.se23 + model.ppa[22] + model.em23)
model.cse23 = pyo.Constraint(expr=model.ce24 == model.se23 - model.se24 + model.ppa[23] + model.em24)

model.csg1 = pyo.Constraint(expr=(model.sg2-model.sg1)*eff_el == model.ce2 - demand)
model.csg2 = pyo.Constraint(expr=(model.sg3-model.sg2)*eff_el == model.ce3 - demand)
model.csg3 = pyo.Constraint(expr=(model.sg4-model.sg3)*eff_el == model.ce4 - demand)
model.csg4 = pyo.Constraint(expr=(model.sg5-model.sg4)*eff_el == model.ce5 - demand)
model.csg5 = pyo.Constraint(expr=(model.sg6-model.sg5)*eff_el == model.ce6 - demand)
model.csg6 = pyo.Constraint(expr=(model.sg7-model.sg6)*eff_el == model.ce7 - demand)
model.csg7 = pyo.Constraint(expr=(model.sg8-model.sg7)*eff_el == model.ce8 - demand)
model.csg8 = pyo.Constraint(expr=(model.sg9-model.sg8)*eff_el == model.ce9 - demand)
model.csg9 = pyo.Constraint(expr=(model.sg10-model.sg9)*eff_el == model.ce10 - demand)
model.csg10 = pyo.Constraint(expr=(model.sg11-model.sg10)*eff_el == model.ce11 - demand)
model.csg11 = pyo.Constraint(expr=(model.sg12-model.sg11)*eff_el == model.ce12 - demand)
model.csg12 = pyo.Constraint(expr=(model.sg13-model.sg12)*eff_el == model.ce13 - demand)
model.csg13 = pyo.Constraint(expr=(model.sg14-model.sg13)*eff_el == model.ce14 - demand)
model.csg14 = pyo.Constraint(expr=(model.sg15-model.sg14)*eff_el == model.ce15 - demand)
model.csg15 = pyo.Constraint(expr=(model.sg16-model.sg15)*eff_el == model.ce16 - demand)
model.csg16 = pyo.Constraint(expr=(model.sg17-model.sg16)*eff_el == model.ce17 - demand)
model.csg17 = pyo.Constraint(expr=(model.sg18-model.sg17)*eff_el == model.ce18 - demand)
model.csg18 = pyo.Constraint(expr=(model.sg19-model.sg18)*eff_el == model.ce19 - demand)
model.csg19 = pyo.Constraint(expr=(model.sg20-model.sg19)*eff_el == model.ce20 - demand)
model.csg20 = pyo.Constraint(expr=(model.sg21-model.sg20)*eff_el == model.ce21 - demand)
model.csg21 = pyo.Constraint(expr=(model.sg22-model.sg21)*eff_el == model.ce22 - demand)
model.csg22 = pyo.Constraint(expr=(model.sg23-model.sg22)*eff_el == model.ce23 - demand)
model.csg23 = pyo.Constraint(expr=(model.sg24-model.sg23)*eff_el == model.ce24 - demand)

# === Specify which parameters are uncertain ===
uncertain_parameters = [model.ppa]


# === Construct the desirable uncertainty set ===
bounds = [(infbound[t], supbound[t]) for t in range(24)]
box_uncertainty_set = pyros.BoxSet(bounds=bounds)


# === Designate local and global NLP solvers ===
local_solver = pyo.SolverFactory('gurobi')
global_solver = pyo.SolverFactory('gurobi')

# === Designate which variables correspond to first-stage
#     and second-stage degrees of freedom ===
first_stage_variables = [
    model.sg1, model.sg2, model.sg3, model.sg4, model.sg5, model.sg6, model.sg7, model.sg8, model.sg9, model.sg10, model.sg11, model.sg12,
    model.sg13, model.sg14, model.sg15, model.sg16, model.sg17, model.sg18, model.sg19, model.sg20, model.sg21, model.sg22, model.sg23, model.sg24,
    model.se1, model.se2, model.se3, model.se4, model.se5, model.se6, model.se7, model.se8, model.se9, model.se10, model.se11, model.se12,
    model.se13, model.se14, model.se15, model.se16, model.se17, model.se18, model.se19, model.se20, model.se21, model.se22, model.se23, model.se24,
    model.ce1, model.ce2, model.ce3, model.ce4, model.ce5, model.ce6, model.ce7, model.ce8, model.ce9, model.ce10, model.ce11, model.ce12,
    model.ce13, model.ce14, model.ce15, model.ce16, model.ce17, model.ce18, model.ce19, model.ce20, model.ce21, model.ce22, model.ce23, model.ce24,
    model.em1, model.em2, model.em3, model.em4, model.em5, model.em6, model.em7, model.em8, model.em9, model.em10, model.em11, model.em12,
    model.em13, model.em14, model.em15, model.em16, model.em17, model.em18, model.em19, model.em20, model.em21, model.em22, model.em23, model.em24
]
second_stage_variables = []

# === Call PyROS to solve the robust optimization problem ===
results_1 = pyros_solver.solve(
    model=model,
    first_stage_variables=first_stage_variables,
    second_stage_variables=second_stage_variables,
    uncertain_params=uncertain_parameters,
    uncertainty_set=box_uncertainty_set,
    local_solver=local_solver,
    global_solver=global_solver,
    objective_focus=pyros.ObjectiveType.worst_case,
    solve_master_globally=True,
    load_solution=False,
)

# === Query results ===
print(results_1.time)
print(results_1.iterations)
print(results_1.pyros_termination_condition)
print(results_1.final_objective_value)

# print("energy marche :")
# for t in range(24):
#     print(model.sg1)

# print("energy ppa")
# for t in range(24):
#     print(model.ppa[t].value)  

# print("stock energy")
# for t in range(24):
#     print(model.stock_energy[t].value) 

# print("stock H2")
# for t in range(24):
#     print(model.stock_gaz[t].value) 