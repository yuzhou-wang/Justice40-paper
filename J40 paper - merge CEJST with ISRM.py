#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import numpy as np

# Merge CEJST with ISRM
## Import the tract level shapefile
Tract_2010_geometry=gpd.read_file("Tract_2010_geometry.shp",index_col = 0)
## Import the ISRM grid shapefile
isrm_polygon = gpd.read_file("isrm_polygon.shp",index_col = 0)

## Import the CEJST data
## Data can be downloaded from https://screeningtool.geoplatform.gov/
justice40v1 = pd.read_csv("Justice40/1.0-communities.csv").rename(columns = {"Census tract 2010 ID":"fips","Identified as disadvantaged":"disadvantaged_v1"})

Tract_2010_merge = Tract_2010_geometry.merge(justice40v1,how = "inner")[["fips","disadvantaged_v1","geometry"]]

## Import the isrm-block crosswalk
isrm_fips_crosswalk = pd.read_csv("isrm_with_fips.csv",index_col = 0)
isrm_fips_crosswalk["fips"] = (isrm_fips_crosswalk["block_fip"]/10000).astype(int)
isrm_tract_j40_crosswalk = isrm_fips_crosswalk.merge(Tract_2010_merge,how = "inner", on = "fips")

## Import the year 2010 census block land area data
## Data can be downloaded from US census
blockpoparea = pd.read_csv('../SpatialDecomposition/US/blockpoparea.csv')[["block_fip","ALAND10"]]
isrm_tract_j40_crosswalk = isrm_tract_j40_crosswalk.merge(blockpoparea)
isrm_tract_j40_crosswalk["ALAND_flag"] = np.where(isrm_tract_j40_crosswalk['disadvantaged_v1']== True, isrm_tract_j40_crosswalk['ALAND10'], 0)
isrm_tract_j40_crosswalk["population_flag"] = np.where(isrm_tract_j40_crosswalk['disadvantaged_v1']== True, isrm_tract_j40_crosswalk['population'], 0)
isrm_flag = isrm_tract_j40_crosswalk.groupby(["isrm_grid_cell_num"]).sum().reset_index()
isrm_flag["population_percentage"] = isrm_flag["population_flag"]/isrm_flag["population"]
isrm_flag["land_percentage"] = isrm_flag["ALAND_flag"]/isrm_flag["ALAND10"]
isrm_flag["flag_j40_v1_land"] = np.where((isrm_flag["population_percentage"]>=0.5)|(isrm_flag["land_percentage"]>=0.5),True,False)

isrm_justice40_crosswalk = isrm_polygon.merge(isrm_flag[["isrm_grid_cell_num","flag_j40_v1_land"]],left_on = "isrm",right_on = "isrm_grid_cell_num",how = "inner")
isrm_justice40_crosswalk.to_csv("isrm_justice40_crosswalk_v1.csv")


# Merge block level race-ethnicity to ISRM
## Import the block level race-ethnicity data
## Data can be downloaded from NHGIS
block_race_ethnicity = pd.read_csv("nhgis0017_ds172_2010_block.csv")
block_race_ethnicity_select = block_race_ethnicity[["GISJOIN","H7Z001","H7Z003","H7Z004","H7Z005","H7Z006","H7Z007","H7Z008",'H7Z009',"H7Z010"]].rename(columns = {"H7Z001":"Population","H7Z003":"White","H7Z004":"Black","H7Z005":"Native","H7Z006":"Asian","H7Z007":"Hawaii","H7Z008":"Other","H7Z009":"Mixed","H7Z010":"Hispanic"})
block_race_ethnicity_select = block_race_ethnicity_select[block_race_ethnicity_select["Population"]>0].reset_index(drop = True)
block_race_ethnicity_select["GISJOIN_blck"] = block_race_ethnicity_select.GISJOIN.str[0:3]+block_race_ethnicity_select.GISJOIN.str[4:7]+block_race_ethnicity_select.GISJOIN.str[8:18]
block_race_ethnicity_select_us = isrm_fips_crosswalk[["isrm_grid_cell_num","GISJOIN_blck"]].merge(block_race_ethnicity_select,how = "left").drop(columns = "GISJOIN")
isrm_pop_race_ethnicity = block_race_ethnicity_select_us.groupby(["isrm_grid_cell_num"]).sum().reset_index().rename(columns = {"isrm_grid_cell_num":"isrm"})
isrm_pop_race_ethnicity.to_csv("isrm_pop_race_ethnicity.csv")

isrm_pop_race_ethnicity_J40_v1_land = isrm_pop_race_ethnicity[isrm_pop_race_ethnicity.isrm.isin(isrm_justice40_crosswalk[isrm_justice40_crosswalk["flag_j40_v1_land"]==True].reset_index(drop = True).isrm.to_list())].reset_index(drop = True)
isrm_pop_race_ethnicity_J40_v1_land.to_csv("isrm_pop_race_ethnicity_J40_v1_land.csv")
isrm_pop_race_ethnicity_outside_v1_land = isrm_pop_race_ethnicity[~isrm_pop_race_ethnicity.isrm.isin(isrm_justice40_crosswalk[isrm_justice40_crosswalk["flag_j40_v1_land"]==True].reset_index(drop = True).isrm.to_list())].reset_index(drop = True)
isrm_pop_race_ethnicity_outside_v1_land.to_csv("isrm_pop_race_ethnicity_outside_v1_land.csv")

