"""
@author: abhay

General description
-------------------
The following energy system is modeled:

                 input/Output    bel
                       |          |           
                       |          | 
                       |          | 
 pv_rt(FixedSource)    |--------->|
                       |          |
 elgrid(FixedSource)   |--------->|
                       |          |
                       |          |
 demand_dsm(SinkDSM)   |<---------|
                       |          |
                       |          |
                     
"""

# Default logger of oemof
from oemof.tools import logger
from oemof import solph
from oemof.tools import economics
from oemof.network.network import Node

import logging
import os
# import po
import pandas as pd
import demandlib.bdew as bdew
import demandlib.particular_profiles as profiles

import pprint as pp


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

    
solver = "cbc"  # 'glpk', 'gurobi',....
debug = False  # Set number_of_timesteps to 3 to get a readable lp-file.
solver_verbose = False  # show/hide solver output

# initiate the logger 
logger.define_logging(
    logfile="dsm_model.log",
    screen_level=logging.INFO,
    file_level=logging.DEBUG,
)
number_of_time_steps = 24 * 365
logging.info("Initialize the energy system")
date_time_index = pd.date_range(
    "2021-01-01", periods= number_of_time_steps, freq="H"
)

es = solph.EnergySystem(timeindex=date_time_index)


ann_el_demand_per_sector = {
    "h0": 1000,
}
year = 2021


# read standard load profiles
e_slp = bdew.ElecSlp(year, holidays=None)

# multiply given annual demand with timeseries
H0_demand = e_slp.get_profile(ann_el_demand_per_sector)

# Resample 15-minute values to hourly values.
H0_demand_hourly = H0_demand.resample("H").mean()


# Read data file
filename = os.path.join(os.getcwd(), "dsm_model.csv")
system_properties = pd.read_csv("system_properties.csv", encoding = 'unicode_escape', index_col = 'index')
Parameter_PV_Wind = pd.read_csv("Parameter_PV_Wind.csv", encoding = 'unicode_escape', sep=';', index_col = 'index', decimal=',')
Feed_in_profiles = pd.read_csv("Feed_in_profiles.csv", encoding = 'unicode_escape', sep=';', decimal=',')

#%%Converting quaterly load profile to hourly

data_feed_in_pv_rt = [None]*number_of_time_steps

i = 0
for a in range(0, number_of_time_steps):
    Sum_data_feed_in_pv_rt = 0
    #Calculate the sum of 4 values in a row
    for k in range(0, 4):
        Sum_data_feed_in_pv_rt += Feed_in_profiles['PV_rt'][i]
        i += 1
        #Make a quarter of the hourly mean value
        data_feed_in_pv_rt[a] = Sum_data_feed_in_pv_rt/4
        
        


#%%Precalculation for Demand Response

H0_max = H0_demand_hourly['h0'].max()
H0_min = H0_demand_hourly['h0'].min()
nominal_value = 3


min_demand = 1*H0_min
length = len(H0_demand_hourly.index)
P_min = pd.Series([1] * length, index = H0_demand_hourly.index)
H0_demand_hourly['P_min'] = P_min
H0_demand_hourly['P_min'] = H0_demand_hourly['P_min'] * min_demand
P_max = pd.Series([1] * length, index = H0_demand_hourly.index)
H0_demand_hourly['P_max'] = P_max * H0_max
H0_demand_hourly['capacity_down'] = (H0_demand_hourly['h0'] - H0_demand_hourly['P_min']).clip(0)
H0_demand_hourly['capacity_up'] = ((H0_demand_hourly['P_max']*0.8) - H0_demand_hourly['h0']).clip(0)
H0_demand_hourly['H0'] = H0_demand_hourly['h0']* nominal_value



   
#%%
##########################################################################
######  Cost Calculation  #################################
##########################################################################


Interest_rate=system_properties['System']['Interest_rate']

def epc_calc(capex, opex, Payback_period):
    investment_cost = economics.annuity(capex=capex, n=Payback_period, wacc=Interest_rate/100)
    operation_cost = capex * (opex/100)
    epc = investment_cost + operation_cost
    
    return epc, investment_cost, operation_cost


# PV rooftop
epc_PV_rt, investment_cost_PV_rt, operation_cost_PV_rt = epc_calc(Parameter_PV_Wind['PV_rt']['CAPEX'], 
                                                                  Parameter_PV_Wind['PV_rt']['OPEX'], 
                                                                  Parameter_PV_Wind['PV_rt']['Payback_period']
                                                                  )

#%%
##########################################################################
# Create oemof object
##########################################################################

logging.info("Create oemof objects")


# create electricity bus
b_el = solph.Bus(label="electricity")


# adding the buses to the energy system
es.add(b_el)


# create excess component for the electricity bus to allow overproduction
es.add(solph.Sink(label="excess_el", inputs={b_el: solph.Flow(variable_costs=1000000)}))
#es.add(solph.Sink(label = "excess_del", inputs = {b_del: solph.Flow(variable_costs=1000000)}))


#%%Residential Sector

# create fixed source object representing electricity grid
es.add(
    solph.Source(
        label="elgrid",
        outputs={b_el: solph.Flow(variable_costs = 0.4804
                                 )},
    )
)


# create fixed source object representing pv rooftop
es.add(
    solph.Source(
        label="pv_rt",
        outputs={b_el: solph.Flow(fix=data_feed_in_pv_rt,
                                 investment=solph.Investment(ep_costs=epc_PV_rt, maximum = 20),
                                 )},
    )
)


# create SinkDSM component to implement demand side management
demand_dsm = solph.custom.SinkDSM(
           label = 'demand_dsm',
           demand = H0_demand_hourly["h0"],
           capacity_up = H0_demand_hourly['capacity_up'],
           capacity_down = H0_demand_hourly['capacity_down'],
           approach = "DLR",
           inputs = {b_el : solph.Flow()},
           delay_time = 24,
           cost_dsm_up = 0,
           cost_dsm_down_shift = 0,
           shed_eligibility = False,
           max_demand = nominal_value,
           max_capacity_up = nominal_value,
           max_capacity_down = nominal_value,
           shift_time = 6,
           )

es.add(demand_dsm)


#%%
##########################################################################
# Optimise the energy system and plot the results
##########################################################################

logging.info("Optimise the energy system")
model = solph.Model(es)


# This is for debugging only. It is not(!) necessary to solve the problem and
# should be set to False to save time and disc space in normal use. For
# debugging the timesteps should be set to 3, to increase the readability of
# the lp-file.
if debug:
    filename = os.path.join(
        solph.helpers.extend_basic_path('lp_files'), 'dsm_model.lp')
    logging.info('Store lp-file in {0}.'.format(filename))
    model.write(filename, io_options={'symbolic_solver_labels': True})



# if tee_switch is true solver messages will be displayed
logging.info("Solve the optimization problem")
model.solve(solver=solver, solve_kwargs={"tee": solver_verbose})


logging.info("Store the energy system with the results.")

# The processing module of the outputlib can be used to extract the results
# from the model transfer them into a homogeneous structured dictionary.

# add results to the energy system to make it possible to store them.
es.results["main"] = solph.processing.results(model)
es.results["meta"] = solph.processing.meta_results(model)

# define an alias for shorter calls below (optional)
results = es.results["main"]


logging.info("Exporting results to excel")

# get all variables of a specific component/bus
electricity_bus = solph.views.node(results, "electricity")
Sink_dsm = solph.views.node(results, "demand_dsm")

#%%
# Exporting the results of each buses to excel
electricity_bus['sequences'].to_excel(r'Results\electricity_bus_results.xlsx', index = True, header= True)
Sink_dsm['sequences'].to_excel(r'Results\Sink_dsm_results.xlsx', index = True, header= True)
H0_demand_hourly.to_excel(r'Results\H0_demand_hourly.xlsx', index = True, header= True)

#%%
#Results Extraction

Demand_results = pd.DataFrame(data= None, index = None)
Demand_results['Demand_dsm'] = electricity_bus["sequences"][(('electricity', 'demand_dsm'), 'flow')]
Demand_results['H0_demand'] = H0_demand_hourly['H0']
Demand_results['Power_shifted'] = (Demand_results['H0_demand'] - Demand_results['Demand_dsm']).clip(0)

#%%

# plot the time series (sequences) of a specific component/bus
if plt is not None:
    fig, ax = plt.subplots(figsize=(10, 5))
    electricity_bus["sequences"].plot(
        ax=ax, kind="line", drawstyle="steps-post"
    )
    plt.legend(
        loc="upper center", prop={"size": 8}, bbox_to_anchor=(0.5, 1.3), ncol=2
    )
    fig.subplots_adjust(top=0.8)
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 5))
    electricity_bus["sequences"][(('electricity', 'demand_dsm'), 'flow')].plot(
        ax=ax, kind="line", drawstyle="default"
    )
    H0_demand_hourly['H0'].plot(
        ax=ax, kind="line", drawstyle="default")
    plt.legend(
        loc="upper center", prop={"size": 8}, bbox_to_anchor=(0.5, 1.3), ncol=2
    )
    fig.subplots_adjust(top=0.8)
    plt.show()


# Plot demand
ax = H0_demand_hourly.plot()
ax.set_xlabel("Date")
ax.set_ylabel("Power demand")
plt.show()



# print the solver results
print("********* Meta results *********")
pp.pprint(es.results["meta"])
print("")
#%%
# print the sums of the flows around the electricity bus
print("********* Main results *********")
print(electricity_bus["sequences"].sum(axis=0))


#%% Electricity Import Cost Calculation
Import_cost_el = 0

for i in range(0, len(electricity_bus['sequences'][('elgrid','electricity'),'flow'])):
   Import_cost_el += (electricity_bus['sequences'][('elgrid','electricity'),'flow'][i]) * (0.4804) 


#%% PV Electricity Cost Calculation

#Investment cost

CAPEX_Pv_rt = (electricity_bus['scalars'][(('pv_rt', 'electricity'), 'invest')]) * investment_cost_PV_rt

# Operation Cost 

OPEX_Pv_rt = (electricity_bus['scalars'][(('pv_rt', 'electricity'), 'invest')]) * operation_cost_PV_rt



#%% Total electricity Cost

Total_cost_el = Import_cost_el + CAPEX_Pv_rt + OPEX_Pv_rt 
Demand_results["Cost"] = Total_cost_el
Demand_results.to_excel(r'Results\Demand_results.xlsx', index = True, header= True)

Feed_in = pd.DataFrame(data = None, index = None)
Feed_in['PV_rt'] = data_feed_in_pv_rt
Feed_in.to_excel(r'Results\Feed_in_pv.xlsx')



#%%
#Daily data conversion
Demand_results_daily = Demand_results.resample('D').mean()
Demand_results_daily.to_excel(r'Results\Demand_results_daily.xlsx',index =True, header=True)

#%%
#Printing Results

print("")
print("******Total Cost of the Model******")
pp.pprint(Total_cost_el)
print("")
print("******Total Power Shifted******")
print(Demand_results['Power_shifted'].sum(axis=0))
print(Demand_results['Demand_dsm'].sum(axis=0))
print(Demand_results['H0_demand'].sum(axis=0))
