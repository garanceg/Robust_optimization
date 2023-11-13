import pyomo.environ as pyo
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import time

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

# model = pyo.ConcreteModel()

# Données du probleme
# boundppa = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 30, 20, 10, 0, 50]

prix_ppa = 20
# prix_marche = [20.79, 17.41, 16.24, 11.9, 9.77, 15.88, 24.88, 29.7, 35.01, 33.95, 29.9, 29.03, 27.07, 26.43, 27.53, 29.05, 31.42, 39.92, 41.3, 41.51, 39.75, 30.13, 30.36, 32.4][::-1]
prix_marche = [20.79, 17.41, 16.24, 11.9, 9.77, 15.88, 24.88, 29.7, 35.01, 33.95, 29.9, 29.03, 27.07, 26.43, 27.53, 29.05, 31.42, 39.92, 41.3, 100, 100, 100, 100, 100][::-1]
print(prix_marche)
capex_electrolyseur = 1200000*0.0004/100
capex_batterie = 250000*0.0002/100
capex_stock_gaz = 407*0.000137 
# capex_stock_gaz = 0.05
eff_el = 0.05
demand = mean_mrf_tot*1.5


def model_plne_robust(stock_gaz, stock_energy, stock_el):
    model = pyo.ConcreteModel()

    # Variables du probleme
    model.T = pyo.RangeSet(0,23)
    model.T_sup = pyo.RangeSet(1,23)
    model.stock_gaz = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, stock_gaz) for _ in model.T])
    model.stock_energy = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, stock_energy) for _ in model.T])
    model.cap_el = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, stock_el) for _ in model.T])
    # model.ppa = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, boundppa[t]) for t in model.T])
    model.energymarche = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, 10000) for _ in model.T])
    
    model.z = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.mu_z_sup = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.mu_z_inf = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    
    model.mu_demand = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.mu_conserv_el_sup = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.mu_conserv_el_inf = pyo.Var(model.T, domain=pyo.NonNegativeReals)


    # Fonction Objectif
    model.value = pyo.Objective(
        expr=sum(capex_electrolyseur * model.cap_el[t] + capex_batterie * model.stock_energy[t] + capex_stock_gaz * model.stock_gaz[t]*eff_el 
                + prix_marche[t] * model.energymarche[t] + prix_ppa * model.z[t] for t in model.T))


    # Conditions initiales
    def ci_stock_gaz(m):
        return m.stock_gaz[0] == 0

    model.CIStockGaz = pyo.Constraint(rule = ci_stock_gaz)

    def ci_stock_energy(m):
        return m.stock_energy[0] == 0

    model.CIStockEnergy = pyo.Constraint(rule = ci_stock_energy)

    def ci_cap_el(m):
        return m.cap_el[0] == 0

    model.CICap_el = pyo.Constraint(rule=ci_cap_el)


    # Contraintes

    # # (1) Toute la PPA est utilisée
    # def ppa_constraint(m, t):
    #     return m.ppa[t] == boundppa[t]

    # model.PPAConstraint = pyo.Constraint(model.T, rule = ppa_constraint)

    ## (0) contrainte sur les z_t

    def z_constraint_sup(m, t):
        return m.mu_z_sup[t]*(infbound[t]-supbound[t]) + supbound[t] >= m.z[t]
    
    model.ZSupConstraint = pyo.Constraint(model.T, rule = z_constraint_sup)
    
    def z_constraint_inf(m, t):
        return m.mu_z_inf[t]*(supbound[t]-infbound[t]) + supbound[t] <= m.z[t]
    
    model.ZInfConstraint = pyo.Constraint(model.T, rule = z_constraint_inf)


    # (2) La demande est satisfaite
    def demand_constraint(m, t):
        return m.stock_energy[t] + m.mu_demand[t]*(infbound[t]-supbound[t]) + supbound[t] + m.energymarche[t] + m.stock_gaz[t]*eff_el >= demand
    
    model.DemandConstraint = pyo.Constraint(model.T, rule = demand_constraint)

    # (3) Conservation énergie électrolyseur
    def cap_el_constraint_sup(m, t):
        return m.stock_energy[t-1] - m.stock_energy[t] + m.mu_conserv_el_sup[t]*(infbound[t]-supbound[t]) + supbound[t] + m.energymarche[t] >= m.cap_el[t]
    
    model.CapSupElConstraint = pyo.Constraint(model.T_sup, rule = cap_el_constraint_sup)

    def cap_el_constraint_inf(m, t): 
        return m.stock_energy[t-1] - m.stock_energy[t] + m.mu_conserv_el_inf[t]*(supbound[t]-infbound[t]) + supbound[t] + m.energymarche[t] <= m.cap_el[t]
    
    model.CapInfElConstraint = pyo.Constraint(model.T_sup, rule = cap_el_constraint_inf)

    # (4) Répartission du stock gaz
    def stock_gaz_constraint(m, t):
        return (m.stock_gaz[t] - m.stock_gaz[t-1])*eff_el == m.cap_el[t] - demand
        # return (m.stock_gaz[t] - m.stock_gaz[t-1])*eff_el == m.cap_el[t] - demand

    model.StockGazConstraint = pyo.Constraint(model.T_sup, rule = stock_gaz_constraint)


    opt = pyo.SolverFactory('gurobi')
    result_obj = opt.solve(model)
    model.pprint()
    print(result_obj)

    x = [t for t in range(24)]

    y_1 = [model.energymarche[t].value for t in range(24)]
    y_2 = [model.z[t].value for t in range(24)]

    fig_1, ax_1 = plt.subplots()

    ax_1.plot(x, y_1, label='energie marché')
    ax_1.plot(x, y_2, label='ppa')
    ax_1.plot(x, infbound, '.', label="inf ppa")
    ax_1.plot(x, supbound, '.', label="sup ppa")

    # Add a legend
    ax_1.legend()

    # Add labels and title
    ax_1.set_xlabel('hour')
    ax_1.set_title('courbes')

    # Display the plot
    plt.show()

    y_3 = [model.stock_energy[t].value for t in range(24)]
    y_4 = [model.stock_gaz[t].value*0.05 for t in range(24)]

    fig_2, ax_2 = plt.subplots()

    ax_2.plot(x, y_3, label='stock energie')
    ax_2.plot(x, y_4, label='stock gaz')

    # Add a legend
    ax_2.legend()

    # Add labels and title
    ax_2.set_xlabel('hour')
    ax_1.set_title('courbes')

    # Display the plot
    plt.show()

    return model.value()

start_time = time.time()

val = model_plne_robust(3800, 650, 450)

# Fin du chronomètre
end_time = time.time()

# Calcul du temps écoulé
execution_time = end_time - start_time


print(val)
print(execution_time)