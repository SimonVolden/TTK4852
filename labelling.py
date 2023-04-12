import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box


def normalise(pic):
    '''
    Input:
    pic: array of values we want to normalise
    
    Output: 
    new: normalised array
    '''
    max = pic.max()
    min = pic.min()
    new = (pic - min)/(max-min)
    return new

def create_geopandas(data):
    '''
    Input:
    data: Pandas with corrected bands and lat and lon coord per row

    Output:
    Geodataframe, where each row represents a pixel. band values and geometry.
    '''
    df_all = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat))
    df_all["label"] = np.zeros(len(data))
    return df_all

def label(number_of_slices, df_all, seaweed_mask): 
    '''
    Input: 
        number_of_slices: How many slices are we dividing the prisma area into. 
        df_all: Band data from prisma 
        seaweed_mask: Naturdirektoratet data
    Output:
        modifies the input df_all to include a label, True if there is seaweed there and False if there is no seaweed

    '''
    lon_min, lat_min, lon_max, lat_max = df_all.total_bounds
    dlon = (lon_max-lon_min)/number_of_slices
    dlat = (lat_max-lat_min)/number_of_slices
    
    count = 0 
    print("count will go up to " + str(number_of_slices**2))
    for i in range(number_of_slices-1):
        lat_lower = lat_min + i*dlat
        lat_upper = lat_min + (i+1)*dlat
        for j in range(number_of_slices-1):
            lon_left = lon_min + j*dlon
            lon_right = lon_min + (j+1)*dlon
            seaweed_current = gpd.clip(seaweed_mask, box(*(lon_left, lat_lower, lon_right, lat_upper)))
            print(count)
            count+=1
            if not seaweed_current.empty:
                polygons = unary_union(seaweed_current.geometry)
                df_all_current = gpd.clip(df_all, box(*(lon_left, lat_lower, lon_right, lat_upper)))
                df_all.loc[df_all_current.index, "label"] = df_all_current.geometry.within(polygons).astype(int)


#filename contains a pickled pandas dataframe of the following format: 
#66 band values + lat + lon - In all 68 columns
filename = "..\data\df_for_malan.pickle"
data = pd.read_pickle(filename)

print(f"Datafile {filename} has been read.")


#The file "../data/naturdirektoratet.json" has information about seaweed in Norway
data_naturdirektoratet = gpd.read_file("../data/naturdirektoratet.json")
seaweed = data_naturdirektoratet.copy()

#We are only interested in the seaweed 
seaweed= seaweed.loc[seaweed["naturtype"] == "størreTareskogforekomster"]


print("The data from NaturDirektoratet has been read.")
#Creating a geopandas from the prisma data
df_all = create_geopandas(data)

#Only including the data from Naturdirektoratet which is within the prisma area.
seaweed_mask= gpd.clip(seaweed, box(*df_all.total_bounds))

#labeling the data
number_of_slice = 10                   
label(number_of_slice, df_all, seaweed)
print("The prisma data has been labelled.")


#Creating a plot to see wheter the method is labelling correctly 
#Note red = prisma data, blue = Naturdirektoratet data
df_all_true = df_all[df_all["label"] == True].copy()
fig, ax = plt.subplots()
df_all_true.plot(ax = ax, color = "blue")
seaweed_mask.plot(ax = ax, color = "red")
ax.set_title("Labelling Prisma data")
ax.set_xlabel("lon")
ax.set_ylabel("lat")
plt.savefig("labelled_prisma.jpg")

#Saving the data
df_all.to_csv("..\data\tare_felt_1.csv")