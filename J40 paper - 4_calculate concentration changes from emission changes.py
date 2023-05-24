#!/usr/bin/env python
# coding: utf-8
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
import seaborn as sns
from netCDF4 import Dataset
import numpy.ma as ma
from statsmodels.stats.weightstats import DescrStatsW

# Combine CEJST, EJScreen, and Census race-ethnicity data
## import isrm & census block crosswalk
isrm_fips_crosswalk = pd.read_csv("isrm_with_fips.csv",index_col = 0)

## import ejscreen data
## data can be downloaded from EJSCREEN
EJSCREEN_2020 = pd.read_csv("Justice40/EJSCREEN_2020_USPR.csv")["ID","MINORPCT","LOWINCPCT","PM25","P_MINORPCT","P_LWINCPCT","P_PM25"]]

isrm_fips_crosswalk["bg_ID"] = (isrm_fips_crosswalk["block_fip"]/1000).astype(int)
isrm_bg_crosswalk = isrm_fips_crosswalk.groupby(["isrm_grid_cell_num","bg_ID"]).agg({"population":"sum","centroid_latitude":"mean","centroid_longitude":"mean"}).reset_index()
isrm_bg_crosswalk = isrm_bg_crosswalk.rename(columns = {"isrm_grid_cell_num":"isrm","bg_ID":"ID"}).merge(EJSCREEN_2020,how = "left")

isrm_ejscreen = isrm_bg_crosswalk.groupby(["isrm"]).count().reset_index()[["isrm"]]
for i in isrm_bg_crosswalk.columns[5:13]:
    isrm_bg_crosswalk["temp"] = isrm_bg_crosswalk[i]*isrm_bg_crosswalk["population"]
    g = isrm_bg_crosswalk.groupby(["isrm"])
    isrm_ejscreen[i]= (g["temp"].sum()/g["population"].sum()).values

## import race-ethnicity data
isrm_pop_race_ethnicity = pd.read_csv("isrm_pop_race_ethnicity.csv",index_col = 0)

## merge ejscreen data with race-ethnicity data
isrm_multiyear_ejscreen = isrm_pop_race_ethnicity.merge(isrm_ejscreen)

## import justice40 race-ethnicity data
isrm_pop_race_ethnicity_J40_v1_land=pd.read_csv("isrm_pop_race_ethnicity_J40_v1_land.csv",index_col = 0)
isrm_pop_race_ethnicity_outside_v1_land=pd.read_csv("isrm_pop_race_ethnicity_outside_v1_land.csv",index_col = 0)

## merge three data sets
isrm_multiyear_ejscreen["J40_v1_land"] = np.where(isrm_multiyear_ejscreen.isrm.isin(isrm_pop_race_ethnicity_J40_v1_land.isrm.to_list()),"T","F")
isrm_multiyear_ejscreen["Other_Mixed"] = isrm_multiyear_ejscreen[["Other","Mixed"]].sum(axis = 1)
isrm_multiyear_ejscreen["Asian_Pacific"] = isrm_multiyear_ejscreen[["Hawaii","Asian"]].sum(axis = 1)
isrm_multiyear_ejscreen["Poverty"] = isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"]
isrm_multiyear_ejscreen["Non-Poverty"] = isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"])
isrm_multiyear_ejscreen["POC"] = isrm_multiyear_ejscreen["Population"]-isrm_multiyear_ejscreen["White"]
isrm_multiyear_ejscreen_J40 = isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["J40_v1_land"]=="T"]
isrm_multiyear_ejscreen_Outside = isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["J40_v1_land"]=="F"]
isrm_multiyear_ejscreen.to_csv("isrm_multiyear_ejscreen.csv")

## Calculate demographic composition for Justice40 and non- Justice40 locations
isrm_multiyear_ejscreen_J40 = isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["J40_v1_land"]=="T"]
isrm_multiyear_ejscreen_Outside = isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["J40_v1_land"]=="F"]

### race-ethnicity
race_composition_list_Total = []
for race in ["White","Hispanic","Black","Asian_Pacific","Native","Other_Mixed"]:
    race_composition_list_Total.append(isrm_multiyear_ejscreen[race].sum()/isrm_multiyear_ejscreen["Population"].sum()*100)   

race_composition_list_J40 = []
for race in ["White","Hispanic","Black","Asian_Pacific","Native","Other_Mixed"]:
    race_composition_list_J40.append(isrm_multiyear_ejscreen_J40[race].sum()/isrm_multiyear_ejscreen_J40["Population"].sum()*100)

race_composition_list_Outside = []
for race in ["White","Hispanic","Black","Asian_Pacific","Native","Other_Mixed"]:
    race_composition_list_Outside.append(isrm_multiyear_ejscreen_Outside[race].sum()/isrm_multiyear_ejscreen_Outside["Population"].sum()*100)
    race_ethnic_composition_j40 = pd.DataFrame([race_composition_list_Total,race_composition_list_J40,race_composition_list_Outside],index=['Overall (whole US)',
                               'Justice40 communities',
                               'Other communities'],columns = ["White","Hispanic","Black","Asian","Native","Other/Mixed"])

race_composition_list_POC_Total = []
for race in ["Hispanic","Black","Asian_Pacific","Native","Other_Mixed"]:
    race_composition_list_POC_Total.append(isrm_multiyear_ejscreen[race].sum()/(isrm_multiyear_ejscreen["Population"].sum()-isrm_multiyear_ejscreen["White"].sum())*100)

race_composition_list_POC_J40 = []
for race in ["Hispanic","Black","Asian_Pacific","Native","Other_Mixed"]:
    race_composition_list_POC_J40.append(isrm_multiyear_ejscreen_J40[race].sum()/(isrm_multiyear_ejscreen_J40["Population"].sum()-isrm_multiyear_ejscreen_J40["White"].sum())*100)

race_composition_list_POC_Outside = []
for race in ["Hispanic","Black","Asian_Pacific","Native","Other_Mixed"]:
    race_composition_list_POC_Outside.append(isrm_multiyear_ejscreen_Outside[race].sum()/(isrm_multiyear_ejscreen_Outside["Population"].sum()-isrm_multiyear_ejscreen_Outside["White"].sum())*100)

race_ethnic_composition_POC_j40 = pd.DataFrame([race_composition_list_POC_Total,race_composition_list_POC_J40,race_composition_list_POC_Outside],index=['Overall (whole US)',
                               'Justice40 communities',
                               'Other communities'],columns = ["Hispanic","Black","Asian","Native","Other/Mixed"])#.reset_index()#.rename(columns = {"index":"Race"})

J40_composition_list_White = [isrm_multiyear_ejscreen_J40.White.sum()/isrm_multiyear_ejscreen.White.sum()*100,
                             isrm_multiyear_ejscreen_Outside.White.sum()/isrm_multiyear_ejscreen.White.sum()*100]
J40_composition_list_Hispanic = [isrm_multiyear_ejscreen_J40["Hispanic"].sum()/isrm_multiyear_ejscreen["Hispanic"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["Hispanic"].sum()/isrm_multiyear_ejscreen["Hispanic"].sum()*100]
J40_composition_list_Black = [isrm_multiyear_ejscreen_J40["Black"].sum()/isrm_multiyear_ejscreen["Black"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["Black"].sum()/isrm_multiyear_ejscreen["Black"].sum()*100]
J40_composition_list_Asian = [isrm_multiyear_ejscreen_J40["Asian_Pacific"].sum()/isrm_multiyear_ejscreen["Asian_Pacific"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["Asian_Pacific"].sum()/isrm_multiyear_ejscreen["Asian_Pacific"].sum()*100]
J40_composition_list_Native = [isrm_multiyear_ejscreen_J40["Native"].sum()/isrm_multiyear_ejscreen["Native"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["Native"].sum()/isrm_multiyear_ejscreen["Native"].sum()*100]
J40_composition_list_Other = [isrm_multiyear_ejscreen_J40["Other_Mixed"].sum()/isrm_multiyear_ejscreen["Other_Mixed"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["Other_Mixed"].sum()/isrm_multiyear_ejscreen["Other_Mixed"].sum()*100]
J40_composition_list_POC = [isrm_multiyear_ejscreen_J40["POC"].sum()/isrm_multiyear_ejscreen["POC"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["POC"].sum()/isrm_multiyear_ejscreen["POC"].sum()*100]

J40_race_composition = pd.DataFrame([J40_composition_list_Total,J40_composition_list_White,J40_composition_list_POC,
                                    J40_composition_list_Hispanic,J40_composition_list_Black,J40_composition_list_Asian,
                                    J40_composition_list_Native,J40_composition_list_Other],index=['Overall (whole US)',
                               'White','POC','Hispanic','Black','Asian','Native','Other/Mixed'],columns = ["Justice40 communities","Other communities"])#.reset_index()#.rename(columns = {"index":"poverty"})


### poverty
poverty_composition_list_Total = []
for poverty in ["Poverty","Non-Poverty"]:
    poverty_composition_list_Total.append(isrm_multiyear_ejscreen[poverty].sum()/isrm_multiyear_ejscreen["Population"].sum()*100)
   
poverty_composition_list_J40 = []
for poverty in ["Poverty","Non-Poverty"]:
    poverty_composition_list_J40.append(isrm_multiyear_ejscreen_J40[poverty].sum()/isrm_multiyear_ejscreen_J40["Population"].sum()*100)

poverty_composition_list_Outside = []
for poverty in ["Poverty","Non-Poverty"]:
    poverty_composition_list_Outside.append(isrm_multiyear_ejscreen_Outside[poverty].sum()/isrm_multiyear_ejscreen_Outside["Population"].sum()*100)

poverty_ethnic_composition_j40 = pd.DataFrame([poverty_composition_list_Total,poverty_composition_list_J40,poverty_composition_list_Outside],index=['Overall (whole US)',
                               'Justice40 communities',
                               'Other communities'],columns = ["Low-income population","Other population"])

J40_composition_list_Total = [isrm_multiyear_ejscreen_J40.Population.sum()/isrm_multiyear_ejscreen.Population.sum()*100,
                             isrm_multiyear_ejscreen_Outside.Population.sum()/isrm_multiyear_ejscreen.Population.sum()*100]
J40_composition_list_Poverty = [isrm_multiyear_ejscreen_J40.Poverty.sum()/isrm_multiyear_ejscreen.Poverty.sum()*100,
                             isrm_multiyear_ejscreen_Outside.Poverty.sum()/isrm_multiyear_ejscreen.Poverty.sum()*100]
J40_composition_list_NP = [isrm_multiyear_ejscreen_J40["Non-Poverty"].sum()/isrm_multiyear_ejscreen["Non-Poverty"].sum()*100,
                             isrm_multiyear_ejscreen_Outside["Non-Poverty"].sum()/isrm_multiyear_ejscreen["Non-Poverty"].sum()*100]
J40_poverty_composition = pd.DataFrame([J40_composition_list_Total,J40_composition_list_Poverty,J40_composition_list_NP],index=['Overall (whole US)',
                               'Low-income population',
                               'Other population'],columns = ["Justice40 communities","Other communities"])

## plot demographic compositions
fig,ax = plt.subplots(ncols = 2, nrows = 2, figsize = (11,8),dpi = 300)
race_ethnic_composition_j40[["White","Hispanic","Black","Asian","Native","Other/Mixed"]].plot(kind='barh', stacked=True, color=['brown','forestgreen', 'orange','deepskyblue','hotpink','gold'],ax = ax[0,0])
ax[0,0].set_xlabel('Race-ethnicity composition for total population (%)')
ax[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=False, ncol=3,prop = {"size":9})
ax[0,0].invert_yaxis()
ax[0,0].text(-0.4, 1., "A", transform=ax[0,0].transAxes,size=15, weight='bold')
poverty_ethnic_composition_j40.plot(kind='barh', stacked=True, color=['royalblue', 'chocolate'],ax = ax[0,1])
ax[0,1].set_xlabel('Income composition for total population (%)')
ax[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),fancybox=True, shadow=False, ncol=3,prop = {"size":9})
ax[0,1].invert_yaxis()
ax[0,1].text(-0.4, 1., "B", transform=ax[0,1].transAxes,size=15, weight='bold')
J40_race_composition.plot(kind='barh', stacked=True, color=['limegreen', 'purple'],ax = ax[1,0])
ax[1,0].set_xlabel('Percentage population in Justice40 location (%)')
ax[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),fancybox=True, shadow=False, ncol=3,prop = {"size":9})
ax[1,0].invert_yaxis()
ax[1,0].text(-0.4, 1., "C", transform=ax[1,0].transAxes,size=15, weight='bold')
J40_poverty_composition.plot(kind='barh', stacked=True, color=['limegreen', 'purple'],ax = ax[1,1])
ax[1,1].set_xlabel('Percentage population in Justice40 location (%)')
ax[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),fancybox=True, shadow=False, ncol=3,prop = {"size":9})
ax[1,1].invert_yaxis()
ax[1,1].text(-0.4, 1., "D", transform=ax[1,1].transAxes,size=15, weight='bold')
plt.tight_layout(h_pad = 2)
plt.savefig("demographic_compositions.png")


# Calculate concentration changes based on emission reductions by sector

## Import 2014 nei emission data
nei_isrm_summary_state_new=pd.read_csv("nei_isrm_summary_state_new_scc.csv")
nei_isrm_summary_state_new.rename(columns={'PM25': 'PM2_5'}, inplace=True)


## Emission summary by sector only
sector_list = np.array(['Ag.', 'Coal Elec.', 'Const.', 'Cooking', 'Diesel HD Veh.',
       'Gas LD Veh.', 'Industrial', 'Misc.', 'Non-coal Elec.', 'Offroad',
       'Res. Gas', 'Res. Other', 'Res. Wood', 'Road Dst.'], dtype=object)

for num in range(14):
    for height in ["ground","low","high"]:
        vars()['sector_sum_'+str(num)+"_"+height] = nei_isrm_summary_state_new[(nei_isrm_summary_state_new["sector"] ==sector_list[num])&(nei_isrm_summary_state_new["height_type"] ==height)].reset_index(drop =True)

## Emission summary by sector & CEJST
nei_isrm_summary_state_new["J40_v1"] = np.where(nei_isrm_summary_state_new["isrm"].isin(isrm_pop_race_ethnicity_J40_v1_land.isrm.to_list()),"T","F")
for num in range(14):
    for height in ["ground","low","high"]:
        for J40 in["T","F"]:
            vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40] = nei_isrm_summary_state_new[(nei_isrm_summary_state_new["sector"] ==sector_list[num])&(nei_isrm_summary_state_new["height_type"] ==height)&(nei_isrm_summary_state_new["J40_v1"]==J40)].reset_index(drop =True)

## Emission summary by sector & EJScreen
nei_isrm_summary_state_new["LWINC_30"] = np.where(
    nei_isrm_summary_state_new["isrm"].isin(isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["LOWINCPCT"]>
                                                                   DescrStatsW(data=np.array(isrm_multiyear_ejscreen["LOWINCPCT"]), 
                                                                               weights=isrm_multiyear_ejscreen["Population"]).quantile(probs=0.7,return_pandas=False)[0]].isrm.to_list()),"T","F")
nei_isrm_summary_state_new["MINOR_30"] = np.where(
    nei_isrm_summary_state_new["isrm"].isin(isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["MINORPCT"]>
                                                                   DescrStatsW(data=np.array(isrm_multiyear_ejscreen["MINORPCT"]), 
                                                                               weights=isrm_multiyear_ejscreen["Population"]).quantile(probs=0.7,return_pandas=False)[0]].isrm.to_list()),"T","F")
nei_isrm_summary_state_new["PM25_30"] = np.where(
    nei_isrm_summary_state_new["isrm"].isin(isrm_multiyear_ejscreen[isrm_multiyear_ejscreen["PM25"]>
                                                                   DescrStatsW(data=np.array(isrm_multiyear_ejscreen["PM25"]), 
                                                                               weights=isrm_multiyear_ejscreen["Population"]).quantile(probs=0.7,return_pandas=False)[0]].isrm.to_list()),"T","F")
LEZ_index = ["LWINC_30","MINOR_30","PM25_30"]
for ej_variable in LEZ_index:
    vars()['sector_sum_'+ej_variable] = nei_isrm_summary_state_new.groupby(["sector",ej_variable]).agg({"VOC":"sum","NOx":"sum","NH3":"sum","SOx":"sum","PM2_5":"sum"}).reset_index()
    vars()['sector_sum_'+ej_variable].to_csv('sector_sum_'+ej_variable+".csv")
for num in range(14):
    for height in ["ground","low","high"]:
        for ej_variable in LEZ_index:
            for J40 in["T","F"]:
                vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+J40] = nei_isrm_summary_state_new[(nei_isrm_summary_state_new["sector"] ==sector_list[num])&(nei_isrm_summary_state_new["height_type"] ==height)&(nei_isrm_summary_state_new[ej_variable]==J40)].reset_index(drop =True)

## Calculate concentration changes using ISRM
def sector_conc_value_j40(margin_data,emis_sum,pollutant):
    emis_sum_select = emis_sum[emis_sum[pollutant]>0].reset_index(drop = True)
    emis_isrm = emis_sum_select.isrm.to_list()
    emis_conc_total = emis_sum_select[pollutant].to_list()   
    temp1 = margin_data[emis_isrm,:]
    temp2 = temp1*(np.array([emis_conc_total]).T)
    isrm_conc_sum = temp2.sum(axis = 0)
    return(isrm_conc_sum) 

### PM2.5 
for height_ in ["ground","low","high"]:
    if height == "ground":
        layer = "0"
    elif height == "low":
        layer = "1"
    else:
        layer = "2"
    
    file = './PrimaryPM25L'+layer+'.nc'
    PM_isrm = Dataset(file, mode='r')
    PM_isrm_data = ma.getdata(PM_isrm.variables["PrimaryPM25"][0])
    for num in range(14):
        if vars()['sector_sum_'+str(num)+"_"+height].shape[0]>0:
            if vars()['sector_sum_'+str(num)+"_"+height]["PM2_5"].sum()>0:
                vars()['sector_'+str(num)+"_"+height+"_PM"] = sector_conc_value_j40(PM_isrm_data,vars()['sector_sum_'+str(num)+"_"+height],"PM2_5")
    for num in range(14):
        for J40 in["T","F"]:
            if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40].shape[0]>0:
                if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40]["PM2_5"].sum()>0:
                    vars()['sector_v1_'+str(num)+"_"+height+"_"+J40+"_PM"] = sector_conc_value_j40(PM_isrm_data,vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40],"PM2_5")
    for num in range(14):
        for ej_variable in LEZ_index:
            for TF in["T","F"]:
                if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF].shape[0]>0:
                    if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF]["PM2_5"].sum()>0:
                        vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF+"_PM"] = sector_conc_value_j40(PM_isrm_data,vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF],"PM2_5")
    PM_isrm.close()
    PM_isrm_data=0
    gc.collect()

### NO2 
for height_ in ["ground","low","high"]:
    if height == "ground":
        layer = "0"
    elif height == "low":
        layer = "1"
    else:
        layer = "2"
    
    file = './pNO3L'+layer+'.nc'
    NOx_isrm = Dataset(file, mode='r')
    NOx_isrm_data = ma.getdata(NOx_isrm.variables["pNO3"][0])
    for num in range(14):
        if vars()['sector_sum_'+str(num)+"_"+height].shape[0]>0:
            if vars()['sector_sum_'+str(num)+"_"+height]["NOx"].sum()>0:
                vars()['sector_'+str(num)+"_"+height+"_NOx"] = sector_conc_value_j40(NOx_isrm_data,vars()['sector_sum_'+str(num)+"_"+height],"NOx")
    for num in range(14):
        for J40 in["T","F"]:
            if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40].shape[0]>0:
                if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40]["NOx"].sum()>0:
                    vars()['sector_v1_'+str(num)+"_"+height+"_"+J40+"_NOx"] = sector_conc_value_j40(NOx_isrm_data,vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40],"NOx")
    for num in range(14):
        for ej_variable in LEZ_index:
            for TF in["T","F"]:
                if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF].shape[0]>0:
                    if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF]["NOx"].sum()>0:
                        vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF+"_NOx"] = sector_conc_value_j40(NOx_isrm_data,vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF],"NOx")
    NOx_isrm.close()
    NOx_isrm_data=0
    gc.collect()


### SO2
for height_ in ["ground","low","high"]:
    if height == "ground":
        layer = "0"
    elif height == "low":
        layer = "1"
    else:
        layer = "2"
    
    file = './pSO4L'+layer+'.nc'
    SOx_isrm = Dataset(file, mode='r')
    SOx_isrm_data = ma.getdata(SOx_isrm.variables["pSO4"][0])
    for num in range(14):
        if vars()['sector_sum_'+str(num)+"_"+height].shape[0]>0:
            if vars()['sector_sum_'+str(num)+"_"+height]["SOx"].sum()>0:
                vars()['sector_'+str(num)+"_"+height+"_SOx"] = sector_conc_value_j40(SOx_isrm_data,vars()['sector_sum_'+str(num)+"_"+height],"SOx")
    for num in range(14):
        for J40 in["T","F"]:
            if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40].shape[0]>0:
                if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40]["SOx"].sum()>0:
                    vars()['sector_v1_'+str(num)+"_"+height+"_"+J40+"_SOx"] = sector_conc_value_j40(SOx_isrm_data,vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40],"SOx")
    for num in range(14):
        for ej_variable in LEZ_index:
            for TF in["T","F"]:
                if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF].shape[0]>0:
                    if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF]["SOx"].sum()>0:
                        vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF+"_SOx"] = sector_conc_value_j40(SOx_isrm_data,vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF],"SOx")
    SOx_isrm.close()
    SOx_isrm_data=0
    gc.collect()


### NH3
for height_ in ["ground","low","high"]:
    if height == "ground":
        layer = "0"
    elif height == "low":
        layer = "1"
    else:
        layer = "2"
    
    file = './pNH4L'+layer+'.nc'
    NH3_isrm = Dataset(file, mode='r')
    NH3_isrm_data = ma.getdata(NH3_isrm.variables["pNH4"][0])
    for num in range(14):
        if vars()['sector_sum_'+str(num)+"_"+height].shape[0]>0:
            if vars()['sector_sum_'+str(num)+"_"+height]["NH3"].sum()>0:
                vars()['sector_'+str(num)+"_"+height+"_NH3"] = sector_conc_value_j40(NH3_isrm_data,vars()['sector_sum_'+str(num)+"_"+height],"NH3")
    for num in range(14):
        for J40 in["T","F"]:
            if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40].shape[0]>0:
                if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40]["NH3"].sum()>0:
                    vars()['sector_v1_'+str(num)+"_"+height+"_"+J40+"_NH3"] = sector_conc_value_j40(NH3_isrm_data,vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40],"NH3")
    for num in range(14):
        for ej_variable in LEZ_index:
            for TF in["T","F"]:
                if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF].shape[0]>0:
                    if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF]["NH3"].sum()>0:
                        vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF+"_NH3"] = sector_conc_value_j40(NH3_isrm_data,vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF],"NH3")
    NH3_isrm.close()
    NH3_isrm_data=0
    gc.collect()

### VOC
for height_ in ["ground","low","high"]:
    if height == "ground":
        layer = "0"
    elif height == "low":
        layer = "1"
    else:
        layer = "2"
    
    file = './SOAL'+layer+'.nc'
    VOC_isrm = Dataset(file, mode='r')
    VOC_isrm_data = ma.getdata(VOC_isrm.variables["SOA"][0])
    for num in range(14):
        if vars()['sector_sum_'+str(num)+"_"+height].shape[0]>0:
            if vars()['sector_sum_'+str(num)+"_"+height]["VOC"].sum()>0:
                vars()['sector_'+str(num)+"_"+height+"_VOC"] = sector_conc_value_j40(VOC_isrm_data,vars()['sector_sum_'+str(num)+"_"+height],"VOC")
    for num in range(14):
        for J40 in["T","F"]:
            if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40].shape[0]>0:
                if vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40]["VOC"].sum()>0:
                    vars()['sector_v1_'+str(num)+"_"+height+"_"+J40+"_VOC"] = sector_conc_value_j40(VOC_isrm_data,vars()['sector_sum_v1_'+str(num)+"_"+height+"_"+J40],"VOC")
    for num in range(14):
        for ej_variable in LEZ_index:
            for TF in["T","F"]:
                if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF].shape[0]>0:
                    if vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF]["VOC"].sum()>0:
                        vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF+"_VOC"] = sector_conc_value_j40(VOC_isrm_data,vars()['sector_sum_'+str(num)+"_"+height+"_"+ej_variable+"_"+TF],"VOC")
    VOC_isrm.close()
    VOC_isrm_data=0
    gc.collect()

## Merge calculated concentration changes into dataframes
pollutants = ["VOC","NOx","NH3","SOx","PM2_5"]
### By sector only
for sector_num in range(14):
    vars()['sector_'+str(sector_num)] = pd.DataFrame({"isrm":list(range(52411))})
    for pollutant in pollutants:
        if pollutant == "PM2_5":
            vars()['sector_'+str(sector_num)][pollutant] = 0
            for height in ["ground","low","high"]:
                try:
                    vars()['sector_'+str(sector_num)+"_"+height+"_PM"]
                except:
                    vars()['sector_'+str(sector_num)][pollutant] = vars()['sector_'+str(sector_num)][pollutant]
                else:
                    vars()['sector_'+str(sector_num)][pollutant] = vars()['sector_'+str(sector_num)][pollutant] + vars()['sector_'+str(sector_num)+"_"+height+"_PM"]
        else:
            vars()['sector_'+str(sector_num)][pollutant] = 0
            for height in ["ground","low","high"]:
                try:
                    vars()['sector_'+str(sector_num)+"_"+height+"_"+pollutant]
                except:
                    vars()['sector_'+str(sector_num)][pollutant] = vars()['sector_'+str(sector_num)][pollutant]
                else:
                    vars()['sector_'+str(sector_num)][pollutant] = vars()['sector_'+str(sector_num)][pollutant] + vars()['sector_'+str(sector_num)+"_"+height+"_"+pollutant]

for sector_num in range(14):
    vars()['sector_'+str(sector_num)].to_csv('sector_'+str(sector_num)+".csv")

#### Total concentration
sector_total = sector_0+sector_1+sector_2+sector_3+sector_4+sector_5+sector_6+sector_7+sector_8+sector_9+sector_10+sector_11+sector_12+sector_13
sector_total.isrm = range(52411)
sector_total["Total"] = sector_total[["VOC","NOx","NH3","SOx","PM2_5"]].sum(axis = 1)
sector_total.to_csv("Total_conc_by_pollutant.csv")

### By sector & CEJST
for sector_num in range(14):
    for J40 in["T","F"]:
        vars()['sector_v1_'+str(sector_num)+"_"+J40] = pd.DataFrame({"isrm":list(range(52411))})
        for pollutant in pollutants:
            if pollutant == "PM2_5":
                vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] = 0
                for height in ["ground","low","high"]:
                    try:
                        vars()['sector_v1_'+str(sector_num)+"_"+height+"_"+J40+"_PM"]
                    except:
                        vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] = vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant]
                    else:
                        vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] = vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] + vars()['sector_v1_'+str(sector_num)+"_"+height+"_"+J40+"_PM"]
            else:
                vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] = 0
                for height in ["ground","low","high"]:
                    try:
                        vars()['sector_v1_'+str(sector_num)+"_"+height+"_"+J40+"_"+pollutant]
                    except:
                        vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] = vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant]
                    else:
                        vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] = vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant] + vars()['sector_v1_'+str(sector_num)+"_"+height+"_"+J40+"_"+pollutant]

for sector_num in range(14):
    for J40 in["T","F"]:
        vars()['sector_v1_'+str(sector_num)+"_"+J40].to_csv('sector_v1_'+str(sector_num)+"_"+J40+".csv")

### By sector & EJScreen
for sector_num in range(14):
    for ej_variable in LEZ_index:
        for TF in["T","F"]:
            vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF] = pd.DataFrame({"isrm":list(range(52411))})
            for pollutant in pollutants:
                if pollutant == "PM2_5":
                    vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant] = 0
                    for height in ["ground","low","high"]:
                        try:
                            vars()['sector_sum_'+str(sector_num)+"_"+height+"_"+ej_variable+"_"+TF+"_PM"]
                        except:
                            vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant]  = vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant]
                        else:
                            vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant]  = vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant] + vars()['sector_sum_'+str(sector_num)+"_"+height+"_"+ej_variable+"_"+TF+"_PM"]
                else:
                    vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant] = 0
                    for height in ["ground","low","high"]:
                        try:
                            vars()['sector_sum_'+str(sector_num)+"_"+height+"_"+ej_variable+"_"+TF+"_"+pollutant]
                        except:
                            vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant] = vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant]
                        else:
                            vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant] = vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant] + vars()['sector_sum_'+str(sector_num)+"_"+height+"_"+ej_variable+"_"+TF+"_"+pollutant]

for sector_num in range(14):
    for ej_variable in LEZ_index:
        for TF in["T","F"]:
            vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF].to_csv('./LEZ_conc_margin/sector_'+str(sector_num)+"_"+ej_variable+"_"+TF+".csv")


# Calculate concentration changes for demographic groups
## By sector only
sector_sum = nei_isrm_summary_state_new.groupby(["sector",]).agg({"VOC":"sum","NOx":"sum","NH3":"sum","SOx":"sum","PM2_5":"sum"}).reset_index()
sector_emis_reduce_list=[]
pollutant_list=[]
sector_name_list=[]
sector_emis_list = []
conc_p = []
conc_j40_v1_land=[]
conc_outside_v1_land = []
conc_w=[]
conc_b = []
conc_a = []
conc_h = []
conc_m = []
conc_n = []
conc_o = []
conc_lowinc = []
conc_highinc = []

for sector_num in range(14):
    for pollutant in pollutants:
        if sector_sum[pollutant][sector_num]>0:
            sector_emis = sector_sum[pollutant][sector_num]/31709.79
            sector_conc = vars()['sector_'+str(sector_num)][pollutant]
            
            sector_conc_total = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Population"].values)/isrm_multiyear_ejscreen["Population"].sum()
            sector_conc_j40_v1_land = sum(sector_conc[isrm_pop_race_ethnicity_J40_v1_land.isrm.to_list()]*isrm_pop_race_ethnicity_J40_v1_land["Population"].values)/isrm_pop_race_ethnicity_J40_v1_land["Population"].sum()
            sector_conc_outside_v1_land = sum(sector_conc[isrm_pop_race_ethnicity_outside_v1_land.isrm.to_list()]*isrm_pop_race_ethnicity_outside_v1_land["Population"].values)/isrm_pop_race_ethnicity_outside_v1_land["Population"].sum()
            sector_conc_w = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["White"].values)/isrm_multiyear_ejscreen["White"].sum()
            sector_conc_b = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Black"].values)/isrm_multiyear_ejscreen["Black"].sum()
            sector_conc_a = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Asian_Pacific"].values)/isrm_multiyear_ejscreen["Asian_Pacific"].sum()
            sector_conc_h = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Hispanic"].values)/isrm_multiyear_ejscreen["Hispanic"].sum()
            sector_conc_m = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Mixed"].values)/isrm_multiyear_ejscreen["Mixed"].sum()
            sector_conc_n = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Native"].values)/isrm_multiyear_ejscreen["Native"].sum()
            sector_conc_o = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Other"].values)/isrm_multiyear_ejscreen["Other"].sum()
            sector_conc_lowinc = np.nansum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()].values*isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"])/np.nansum(isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"])
            sector_conc_highinc = np.nansum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()].values*isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"]))/np.nansum(isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"]))

            sector_emis_list.append(sector_emis)
            conc_p.append(sector_conc_total)
            conc_j40_v1_land.append(sector_conc_j40_v1_land)
            conc_outside_v1_land.append(sector_conc_outside_v1_land)            
            conc_w.append(sector_conc_w)
            conc_b.append(sector_conc_b)
            conc_a.append(sector_conc_a)
            conc_h.append(sector_conc_h)
            conc_m.append(sector_conc_m)
            conc_n.append(sector_conc_n)
            conc_o.append(sector_conc_o)            
            conc_lowinc.append(sector_conc_lowinc)            
            conc_highinc.append(sector_conc_highinc)            
            pollutant_list.append(pollutant)
            sector_name_list.append(sector_list[sector_num])

sector_pollutant_justice40_ejscreen_race = pd.DataFrame.from_dict({'emis':sector_emis_list,
"conc_p":conc_p,"pollutant":pollutant_list,
"sector":sector_name_list,"conc_j40_v1_land":conc_j40_v1_land,"conc_outside_v1_land":conc_outside_v1_land,
                          "conc_w":conc_w,"conc_b":conc_b,"conc_a":conc_a,
                          "conc_h":conc_h,"conc_m":conc_m,"conc_n":conc_n,
                          "conc_o":conc_o,"conc_lowinc":conc_lowinc,"conc_highinc":conc_highinc})
sector_pollutant_justice40_ejscreen_race.to_csv("sector_pollutant_justice40_ejscreen_race_new.csv")

## By sector & CEJST
sector_sum_v1_J40 = nei_isrm_summary_state_new.groupby(["sector","J40_v1"]).agg({"VOC":"sum","NOx":"sum","NH3":"sum","SOx":"sum","PM2_5":"sum"}).reset_index()
sector_emis_reduce_list=[]
pollutant_list=[]
sector_name_list=[]
sector_emis_list = []
J40_list = []
conc_p = []
conc_j40_v1_land=[]
conc_outside_v1_land = []
conc_w=[]
conc_b = []
conc_a = []
conc_h = []
conc_m = []
conc_n = []
conc_o = []
conc_lowinc = []
conc_highinc = []

for sector_num in range(14):
    for J40 in["T","F"]:
        sector_sum_v1_select = sector_sum_v1_J40[sector_sum_v1_J40["J40_v1"]==J40].reset_index(drop = True)
        for pollutant in pollutants:
            if sector_sum_v1_select[pollutant][sector_num]>0:
                sector_emis = sector_sum_v1_select[pollutant][sector_num]/31709.79
                sector_conc = vars()['sector_v1_'+str(sector_num)+"_"+J40][pollutant]
                sector_conc_total = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Population"].values)/isrm_multiyear_ejscreen["Population"].sum()
                sector_conc_j40_v1_land = sum(sector_conc[isrm_pop_race_ethnicity_J40_v1_land.isrm.to_list()]*isrm_pop_race_ethnicity_J40_v1_land["Population"].values)/isrm_pop_race_ethnicity_J40_v1_land["Population"].sum()
                sector_conc_outside_v1_land = sum(sector_conc[isrm_pop_race_ethnicity_outside_v1_land.isrm.to_list()]*isrm_pop_race_ethnicity_outside_v1_land["Population"].values)/isrm_pop_race_ethnicity_outside_v1_land["Population"].sum()
                sector_conc_w = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["White"].values)/isrm_multiyear_ejscreen["White"].sum()
                sector_conc_b = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Black"].values)/isrm_multiyear_ejscreen["Black"].sum()
                sector_conc_a = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Asian_Pacific"].values)/isrm_multiyear_ejscreen["Asian_Pacific"].sum()
                sector_conc_h = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Hispanic"].values)/isrm_multiyear_ejscreen["Hispanic"].sum()
                sector_conc_m = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Mixed"].values)/isrm_multiyear_ejscreen["Mixed"].sum()
                sector_conc_n = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Native"].values)/isrm_multiyear_ejscreen["Native"].sum()
                sector_conc_o = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Other"].values)/isrm_multiyear_ejscreen["Other"].sum()
                sector_conc_lowinc = np.nansum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()].values*isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"])/np.nansum(isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"])
                sector_conc_highinc = np.nansum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()].values*isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"]))/np.nansum(isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"]))
                sector_emis_list.append(sector_emis)
                conc_p.append(sector_conc_total)
                conc_j40_v1_land.append(sector_conc_j40_v1_land)
                conc_outside_v1_land.append(sector_conc_outside_v1_land)
                conc_w.append(sector_conc_w)
                conc_b.append(sector_conc_b)
                conc_a.append(sector_conc_a)
                conc_h.append(sector_conc_h)
                conc_m.append(sector_conc_m)
                conc_n.append(sector_conc_n)
                conc_o.append(sector_conc_o)            
                conc_lowinc.append(sector_conc_lowinc)            
                conc_highinc.append(sector_conc_highinc)            
                pollutant_list.append(pollutant)
                J40_list.append(J40)
                sector_name_list.append(sector_list[sector_num])

sector_pollutant_justice40_v1_ejscreen_race_LEZ = pd.DataFrame.from_dict({'emis':sector_emis_list,
"conc_p":conc_p,"pollutant":pollutant_list,"J40_v1":J40_list,
"sector":sector_name_list,"conc_j40_v1_land":conc_j40_v1_land,"conc_outside_v1_land":conc_outside_v1_land,
                          "conc_w":conc_w,"conc_b":conc_b,"conc_a":conc_a,
                          "conc_h":conc_h,"conc_m":conc_m,"conc_n":conc_n,
                          "conc_o":conc_o,"conc_lowinc":conc_lowinc,"conc_highinc":conc_highinc})
sector_pollutant_justice40_v1_ejscreen_race_LEZ.to_csv("sector_pollutant_justice40_v1_ejscreen_race_LEZ_new.csv")


## By sector & EJScreen
for ej_variable in LEZ_index:
    sector_sum_EJ = nei_isrm_summary_state_new.groupby(["sector",ej_variable]).agg({"VOC":"sum","NOx":"sum","NH3":"sum","SOx":"sum","PM2_5":"sum"}).reset_index()
    sector_emis_reduce_list=[]
    pollutant_list=[]
    sector_name_list=[]
    sector_emis_list = []
    TF_list = []
    conc_p = []
    conc_j40_v1_land=[]
    conc_outside_v1_land = []
    conc_w=[]
    conc_b = []
    conc_a = []
    conc_h = []
    conc_m = []
    conc_n = []
    conc_o = []
    conc_lowinc = []
    conc_highinc = []

    for sector_num in range(14):
        for TF in["T","F"]:
            sector_sum_select = sector_sum_EJ[sector_sum_EJ[ej_variable]==TF].reset_index(drop = True)
            for pollutant in pollutants:
                if sector_sum_select[pollutant][sector_num]>0:
                    sector_emis = sector_sum_select[pollutant][sector_num]/31709.79
                    sector_conc = vars()['sector_'+str(sector_num)+"_"+ej_variable+"_"+TF][pollutant]
                    sector_conc_total = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Population"].values)/isrm_multiyear_ejscreen["Population"].sum()
                    sector_conc_j40_v1_land = sum(sector_conc[isrm_pop_race_ethnicity_J40_v1_land.isrm.to_list()]*isrm_pop_race_ethnicity_J40_v1_land["Population"].values)/isrm_pop_race_ethnicity_J40_v1_land["Population"].sum()
                    sector_conc_outside_v1_land = sum(sector_conc[isrm_pop_race_ethnicity_outside_v1_land.isrm.to_list()]*isrm_pop_race_ethnicity_outside_v1_land["Population"].values)/isrm_pop_race_ethnicity_outside_v1_land["Population"].sum()
                    sector_conc_w = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["White"].values)/isrm_multiyear_ejscreen["White"].sum()
                    sector_conc_b = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Black"].values)/isrm_multiyear_ejscreen["Black"].sum()
                    sector_conc_a = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Asian_Pacific"].values)/isrm_multiyear_ejscreen["Asian_Pacific"].sum()
                    sector_conc_h = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Hispanic"].values)/isrm_multiyear_ejscreen["Hispanic"].sum()
                    sector_conc_m = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Mixed"].values)/isrm_multiyear_ejscreen["Mixed"].sum()
                    sector_conc_n = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Native"].values)/isrm_multiyear_ejscreen["Native"].sum()
                    sector_conc_o = sum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()]*isrm_multiyear_ejscreen["Other"].values)/isrm_multiyear_ejscreen["Other"].sum()
                    sector_conc_lowinc = np.nansum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()].values*isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"])/np.nansum(isrm_multiyear_ejscreen["Population"]*isrm_multiyear_ejscreen["LOWINCPCT"])
                    sector_conc_highinc = np.nansum(sector_conc[isrm_multiyear_ejscreen.isrm.to_list()].values*isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"]))/np.nansum(isrm_multiyear_ejscreen["Population"]*(1-isrm_multiyear_ejscreen["LOWINCPCT"]))

                    sector_emis_list.append(sector_emis)
                    conc_p.append(sector_conc_total)
                    conc_j40_v1_land.append(sector_conc_j40_v1_land)
                    conc_outside_v1_land.append(sector_conc_outside_v1_land)
                    conc_w.append(sector_conc_w)
                    conc_b.append(sector_conc_b)
                    conc_a.append(sector_conc_a)
                    conc_h.append(sector_conc_h)
                    conc_m.append(sector_conc_m)
                    conc_n.append(sector_conc_n)
                    conc_o.append(sector_conc_o)            
                    conc_lowinc.append(sector_conc_lowinc)            
                    conc_highinc.append(sector_conc_highinc)            
                    pollutant_list.append(pollutant)
                    TF_list.append(TF)
                    sector_name_list.append(sector_list[sector_num])

    vars()["sector_pollutant_justice40_ejscreen_race_LEZ_"+ej_variable] = pd.DataFrame.from_dict({'emis':sector_emis_list,
    "conc_p":conc_p,"pollutant":pollutant_list,ej_variable:TF_list,
    "sector":sector_name_list,"conc_j40_v1_land":conc_j40_v1_land,"conc_outside_v1_land":conc_outside_v1_land,
                                                        "conc_w":conc_w,"conc_b":conc_b,"conc_a":conc_a,
                                                             "conc_h":conc_h,"conc_m":conc_m,"conc_n":conc_n,
                                                             "conc_o":conc_o,"conc_lowinc":conc_lowinc,"conc_highinc":conc_highinc})
for ej_variable in LEZ_index:
    vars()["sector_pollutant_justice40_ejscreen_race_LEZ_"+ej_variable].to_csv("sector_pollutant_justice40_ejscreen_race_LEZ_"+ej_variable+"_new.csv")
