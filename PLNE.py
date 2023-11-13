import pyomo.environ as pyo
import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

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
print(mean_mrf_tot)

best_param = [0, 0, 0]
best_cost = 1000000

param1 = np.linspace(400, 800, 8)
param2 = np.linspace(3000, 6200, 8)
param3 = np.linspace(200, 600, 8)

costs = np.zeros((len(param1), len(param2), len(param3)))


for i, stock_energie in enumerate(param1):
    for j, stock_gazz in enumerate(param2):
        for k, cap_ell in enumerate(param3):
            model = pyo.ConcreteModel()

            # Données du probleme
            #boundppa = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 30, 20, 10, 0, 50]
            boundppa = most_recent_forecast
            prix_ppa = 20
            prix_marche = [20.79, 17.41, 16.24, 11.9, 9.77, 15.88, 24.88, 29.7, 35.01, 33.95, 29.9, 29.03, 27.07, 26.43, 27.53, 29.05, 31.42, 39.92, 41.3, 41.51, 39.75, 30.13, 30.36, 32.4]
            capex_electrolyseur = 1200000*0.0004/100
            capex_batterie = 250000*0.0002/100
            capex_stock_gaz = 407*0.000137 
            eff_el = 0.05
            demand = mean_mrf_tot


            # Variables du probleme
            model.T = pyo.RangeSet(0,23)
            model.T_sup = pyo.RangeSet(1,23)
            model.stock_gaz = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, stock_gazz) for _ in model.T])
            # 1000
            model.stock_energy = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, stock_energie) for _ in model.T])
            # 
            model.cap_el = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, cap_ell) for _ in model.T])
            # 1000
            model.ppa = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=[(0, boundppa[t]) for t in model.T])
            model.energymarche = pyo.Var(model.T, domain=pyo.NonNegativeReals)


            # Fonction Objectif
            model.value = pyo.Objective(
                expr=sum(capex_electrolyseur * model.cap_el[t] + capex_batterie * model.stock_energy[t] + capex_stock_gaz * model.stock_gaz[t]*eff_el 
                        + prix_marche[t] * model.energymarche[t] + prix_ppa * model.ppa[t] for t in model.T))


            # Conditions initiales
            def ci_stock_gaz(m):
                return m.stock_gaz[0] == 0

            model.CIStockGaz = pyo.Constraint(rule = ci_stock_gaz)

            def ci_stock_energy(m):
                return m.stock_energy[0] == 0

            model.CIStockEnergy = pyo.Constraint(rule = ci_stock_energy)


            # Contraintes

            # (1) Toute la PPA est utilisée
            def ppa_constraint(m, t):
                return m.ppa[t] == boundppa[t]

            model.PPAConstraint = pyo.Constraint(model.T, rule = ppa_constraint)

            # (2) La demande est satisfaite
            def demand_constraint(m, t):
                return m.stock_energy[t] + m.ppa[t] + m.energymarche[t] + m.stock_gaz[t]*eff_el >= demand
            
            model.DemandConstraint = pyo.Constraint(model.T, rule = demand_constraint)

            # (3) Conservation énergie électrolyseur
            def cap_el_constraint(m, t):
                return m.cap_el[t] == m.stock_energy[t-1] - m.stock_energy[t] + m.ppa[t] + m.energymarche[t]

            model.CapElConstraint = pyo.Constraint(model.T_sup, rule = cap_el_constraint)

            # (4) Répartission du stock gaz
            def stock_gaz_constraint(m, t):
                return (m.stock_gaz[t] - m.stock_gaz[t-1])*eff_el == m.cap_el[t] - demand

            # # (5) efficacité batterie
            # def eff_battery(m, t):
            #     return (m.stock_energy[t] == 0.9*m.stock_energy[t-1] )

            model.StockGazConstraint = pyo.Constraint(model.T_sup, rule = stock_gaz_constraint)
            opt = pyo.SolverFactory('gurobi')
            result_obj = opt.solve(model)
            # print(stock_energie, stock_gazz, cap_ell)
            # print(model.value())
           
            costs[i, j, k] = model.value()
            print(i, j, k)
            if model.value() < best_cost:
                m = model
                best_cost = m.value()
                best_param = [stock_energie, stock_gazz, cap_ell]


# print("FONCTION COUT")
# print(sum([capex_electrolyseur * model.cap_el.value + capex_batterie * model.stock_energy.value + capex_stock_gaz*model.stock_gaz.value*eff_el 
#              + prix_marche[t] * model.energymarche[t].value + prix_ppa * model.ppa[t] for t in model.T]))

print(best_cost)
print(best_param)

            #print(result_obj)

# print("energy marche :")
# for t in range(24):
#     print(model.energymarche[t].value)

# print("energy ppa")
# for t in range(24):
#     print(model.ppa[t].value)  

# print("stock energy")
# for t in range(24):
#     print(model.stock_energy[t].value) 

# print("stock H2")
# for t in range(24):
#     print(model.stock_gaz[t].value) 

# print("capacité electrolyseur")
# for t in range(24):
#     print(model.cap_el[t].value)


# # Trouver les indices du coût minimal
min_indices = np.unravel_index(np.argmin(costs), costs.shape)
min_param1 = param1[min_indices[0]]
min_param2 = param2[min_indices[1]]
min_param3 = param3[min_indices[2]]
min_cost = costs[min_indices]

  
# creating a dummy dataset
colo = [costs[i, j, k]]
  
# Créer une liste de couleurs pour les points en fonction des coûts
colors = costs.reshape(-1)  # Utilisez les coûts comme couleur

# Créer la figure et l'axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer les points en 3D avec des couleurs correspondantes aux coûts
param1_grid, param2_grid, param3_grid = np.meshgrid(param1, param2, param3, indexing='ij')
ax.scatter(param1_grid, param2_grid, param3_grid, c=colors, cmap='viridis')

# Marquer le coût minimal
ax.scatter(min_param1, min_param2, min_param3, s=100, c='r', marker='o')
ax.text(min_param1, min_param2, min_param3, 'Min cost', color='r')

# Étiqueter les axes
ax.set_xlabel('Stock energie (en MWh)')
ax.set_ylabel('Stock gaz (en kg)')
ax.set_zlabel('Capacité électrolyseur (en MW)')
ax.set_title('Coût minimal en fonction du dimensionnement')

# Ajouter une barre de couleur
cbar = plt.colorbar(mappable=ax.collections[0])
cbar.set_label('Coût')

# Afficher le graphique
plt.show()


# # # setting color bar

# # color_map = cm.ScalarMappable(cmap=cm.Greens_r)
# # color_map.set_array(colo)


# # # creating the heatmap
# # param1_grid, param2_grid, param3_grid = np.meshgrid(param1, param2, param3, indexing='ij')

# # min_value = np.min(colo)
# # min_index = np.argmin(colo)

# # # Tracer le point minimal en rouge
# # ax.scatter(min_index, min_index, c='red', marker='o', label='Minimum')

# # # # Tracer les points en 3D avec des couleurs correspondantes aux coûts
# # img = ax.scatter(param1_grid, param2_grid, param3_grid, c="green", cmap='viridis')
# # plt.colorbar(color_map)


# # # adding title and labels
# # ax.set_title("3D Heatmap")
# # ax.set_xlabel('X-axis')
# # ax.set_ylabel('Y-axis')
# # ax.set_zlabel('Z-axis')
  
# # # displaying plot
# # plt.scatter(min_param1, min_param2, min_param3, s=50, c='r', marker='o')
# # plt.text(min_param1, min_param2, min_param3, f'Min cost: {min_cost}', color='r')
# plt.show()

# # ###HEAT CUBE
# # # Créer la grille 3D de paramètres
# # param1_grid, param2_grid, param3_grid = np.meshgrid(param1, param2, param3, indexing='ij')

# # # Créer la figure et l'axe 3D
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # # Tracer la surface du heat cube avec des couleurs basées sur les valeurs de coût
# # ax.plot_surface(param1_grid, param2_grid, param3_grid, facecolors=plt.cm.viridis(costs / np.max(costs)), alpha=0.8)

# # # Étiqueter les axes
# # ax.set_xlabel('Param1')
# # ax.set_ylabel('Param2')
# # ax.set_zlabel('Param3')
# # ax.set_title('Heat Cube')

# # # Ajouter une barre de couleur
# # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax, shrink=0.6)
# # cbar.set_label('Coût')

# # # Afficher le graphique
# # plt.show()





# # # x = [t for t in range(24)]

# # # y_1 = [model.energymarche[t].value for t in range(24)]
# # # y_2 = [model.ppa[t].value for t in range(24)]
# # # y_ppa = boundppa
# # # y_3 = [demand for t in range(24)]
# # # y_4 = prix_marche

# # # fig_1, ax_1 = plt.subplots()

# # # ax_1.plot(x, y_1, label='Energie marché')
# # # ax_1.plot(x, y_2, label='PPA')
# # # ax_1.plot(x, y_ppa, label="PPA dispo")
# # # ax_1.plot(x, y_4, label="Prix")

# # # # Add a legend


# # # # Add labels and title
# # # plt.xlabel('Temps en heures')
# # # plt.ylabel("Quantité d'énergie en MWh / Prix du marché en €/MWh")
# # # ax_1.legend()

# # # # Display the plot
# # # plt.title("Evolution de l'achat d'énergie du marché et de PPA en fonction du temps")
# # # plt.show()

# # # y_3 = [model.stock_energy[t].value for t in range(24)]
# # # y_4 = [model.stock_gaz[t].value*0.05 for t in range(24)]

# # # fig_2, ax_2 = plt.subplots()

# # # ax_2.plot(x, y_3, label='Stock energie')
# # # ax_2.plot(x, y_4, label='Stock gaz')

# # # # Add a legend
# # # ax_2.legend()

# # # # Add labels and title
# # # plt.xlabel('Temps en heures')
# # # plt.ylabel("Quantité d'énergie stockée en MWh")

# # # # Display the plot
# # # plt.title("Evolution du stock de gaz et d'électricité en fonction du temps")
# # # plt.show()

