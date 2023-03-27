#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import numpy as np
scc_sector = pd.read_csv("SCC_sector_crosswalk.csv").rename(columns = {"sector classification":"sector"})
scc_sector["sector"]=np.where(scc_sector["sector"]=="Gas HD Veh.","Misc.",
                              np.where(scc_sector["sector"]=="Diesel LD Veh.","Misc.",scc_sector["sector"]))
isrm_geometry = gpd.read_file("isrm_geometry.shp",index_col = 0)

def data_preprocess(mydata):
    mydata["x"] = mydata.geometry.x
    mydata["y"] = mydata.geometry.y
    new_data = mydata.drop(columns=['geometry']).merge(isrm_geometry, how="left", on = ["x","y"]).drop(columns=['geometry',"x","y"])
    new_data = new_data[(new_data["VOC"]>0)|(new_data["NOx"]>0)|(new_data["NH3"]>0)|(new_data["SOx"]>0)|(new_data["PM25"]>0)].reset_index(drop = True)
    new_data.SCC = new_data.SCC.astype(int).astype(str)
    return new_data

def set_height(row):
    if row["Height"] <= 57:
        return "ground"
    elif row["Height"] <= 379:
        return "low"
    else:
        return "high"

def nei_sector_sum(sector_data):
    sector_data = sector_data.assign(height_type=sector_data.apply(set_height, axis=1))
    sector_data = sector_data.merge(scc_sector,on = "SCC",how = "inner")
    sector_data_sum = sector_data.groupby(["sector",'isrm',"height_type"]).agg({'VOC':'sum','NOx':'sum','NH3':'sum','SOx':'sum','PM25':'sum','Height':'mean','Diam':'mean','Temp':'mean','Velocity':'mean'}).reset_index()
    return(sector_data_sum)


# afdust
afdust_nei_isrm = gpd.read_file("2014_emissions/afdust.shp")
afdust_nei_isrm_sum = nei_sector_sum(data_preprocess(afdust_nei_isrm))

# ag
ag_nei_isrm = gpd.read_file("2014_emissions/ag.shp")
ag_nei_isrm_sum = nei_sector_sum(data_preprocess(ag_nei_isrm))

# agfire
agfire_nei_isrm = gpd.read_file("2014_emissions/agfire.shp")
agfire_nei_isrm_sum = nei_sector_sum(data_preprocess(agfire_nei_isrm))

# cmv
cmv_nei_isrm = gpd.read_file("2014_emissions/cmv.shp")
cmv_nei_isrm_sum = nei_sector_sum(data_preprocess(cmv_nei_isrm))

# nonpt
nonpt_nei_isrm = gpd.read_file("2014_emissions/nonpt.shp")
nonpt_nei_isrm_sum = nei_sector_sum(data_preprocess(nonpt_nei_isrm))

# nonroad
nonroad_nei_isrm = gpd.read_file("2014_emissions/nonroad.shp")
nonroad_nei_isrm_sum = nei_sector_sum(data_preprocess(nonroad_nei_isrm))

# np_oilgas
np_oilgas_nei_isrm = gpd.read_file("2014_emissions/np_oilgas.shp")
np_oilgas_nei_isrm_sum = nei_sector_sum(data_preprocess(np_oilgas_nei_isrm))

# onroad
onroad_nei_isrm = gpd.read_file("2014_emissions/onroad.shp")
onroad_nei_isrm_sum = nei_sector_sum(data_preprocess(onroad_nei_isrm))

# pt_oilgas
pt_oilgas_nei_isrm = gpd.read_file("2014_emissions/pt_oilgas.shp")
pt_oilgas_nei_isrm_sum = nei_sector_sum(data_preprocess(pt_oilgas_nei_isrm))

# ptagfire
ptagfire_nei_isrm = gpd.read_file("2014_emissions/ptagfire.shp")
ptagfire_nei_isrm_sum = nei_sector_sum(data_preprocess(ptagfire_nei_isrm))

# ptegu
ptegu_nei_isrm = gpd.read_file("2014_emissions/ptegu.shp")
ptegu_nei_isrm_sum = nei_sector_sum(data_preprocess(ptegu_nei_isrm))

# ptnonipm
ptnonipm_nei_isrm = gpd.read_file("2014_emissions/ptnonipm.shp")
ptnonipm_nei_isrm_sum = nei_sector_sum(data_preprocess(ptnonipm_nei_isrm))

# rail
rail_nei_isrm = gpd.read_file("2014_emissions/rail.shp")
rail_nei_isrm_sum = nei_sector_sum(data_preprocess(rail_nei_isrm))

# rwc
rwc_nei_isrm = gpd.read_file("2014_emissions/rwc.shp")
rwc_nei_isrm_sum = nei_sector_sum(data_preprocess(rwc_nei_isrm))

# Merge all sector
all_sector_nei_isrm = pd.concat([afdust_nei_isrm_sum,ag_nei_isrm_sum,agfire_nei_isrm_sum,cmv_nei_isrm_sum,nonpt_nei_isrm_sum,nonroad_nei_isrm_sum,np_oilgas_nei_isrm_sum,onroad_nei_isrm_sum,pt_oilgas_nei_isrm_sum,ptagfire_nei_isrm_sum,ptegu_nei_isrm_sum,ptnonipm_nei_isrm_sum,rail_nei_isrm_sum,rwc_nei_isrm_sum])
nei_sector_isrm_summary = all_sector_nei_isrm.groupby(['sector','isrm',"height_type"]).agg({'VOC':'sum','NOx':'sum','NH3':'sum','SOx':'sum','PM25':'sum','Height':'mean','Diam':'mean','Temp':'mean','Velocity':'mean'}).reset_index()
us_bondary = gpd.read_file("cb_2014_us_state_500k.shp")
us_bondary_reproj = us_bondary.to_crs(isrm_geometry.crs)
us_bondary_reproj.STATEFP = us_bondary_reproj.STATEFP.astype(int)
isrm_geometry_state = gpd.sjoin(isrm_geometry, us_bondary_reproj[["STATEFP","geometry"]], how='left', op='within')
nei_isrm_summary_state_new = nei_sector_isrm_summary.merge(isrm_geometry_state[["isrm","STATEFP"]],how = "left",on = "isrm")
nei_isrm_summary_state_new = nei_isrm_summary_state_new[nei_isrm_summary_state_new.STATEFP>0]
nei_isrm_summary_state_new.to_csv("nei_isrm_summary_state_new_scc.csv", index = False)

