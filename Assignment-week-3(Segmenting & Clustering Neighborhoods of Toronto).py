#!/usr/bin/env python
# coding: utf-8

# # Assignment-Segmenting and Clustering Neighborhoods in Toronto

# ## Part-1

#  ### Importing Libraries and reading data from Wikipedia

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# In[2]:


my_url= 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
tables = pd.read_html(my_url)


# In[3]:


# Here, tables[0] is a data frame
toronto_data=pd.DataFrame(tables[0]) #transform the data into a pandas dataframe 
toronto_data.columns = [
  'Postalcode',
  'Borough',
  'Neighborhood'
]
toronto_data.head()
#tables[0].head()


# In[4]:


toronto_data = tables[0]
toronto_data.shape


# ### Ignore the rows where Borough is 'Not Assigned'

# In[5]:


toronto_data = toronto_data[toronto_data['Borough'] != "Not assigned"]


# In[6]:


toronto_data['Borough'].unique()


# In[7]:


toronto_data.shape


# In[8]:


toronto_data['Neighborhood'].unique()


# ### Assigning the value of Borough to Neighborhood where Neighborhood = 'Not assigned'

# In[9]:


for i, r in toronto_data.iterrows():
    if r.Neighborhood == 'Not assigned':
        r.Neighborhood = r.Borough


# In[10]:


toronto_data['Neighborhood'].unique()


# In[11]:


print('Total unique Postalcode :', len(toronto_data['Postalcode'].unique()))


# In[12]:


print('Total no. of Neighborhood :',len(toronto_data['Neighborhood']))


# In[13]:


toronto_data.head()


# In[14]:


toronto_data.reset_index(inplace=True, drop=True)
toronto_data.head()


# ### Combining the Neighborhood for the same Postalcode and Borough :

# In[15]:


P_last_value = ''
B_last_value = ''
N_last_value = ''
# i=range(211)

for n, row in toronto_data.iterrows():
    if row.Postalcode == P_last_value and row.Borough == B_last_value:
        row.Neighborhood = N_last_value + ',' + row.Neighborhood
        print('A', n)
        toronto_data.iloc[n-1]['Neighborhood'] = 'Repeating value'  
    P_last_value = row.Postalcode
    B_last_value = row.Borough
    N_last_value = row.Neighborhood
    print('B', n)
   
toronto_data.head()        


# In[16]:


toronto_data.reset_index(inplace=True, drop=True)

toronto_data.head(10) 


# In[17]:


toronto_data.shape


# ### Removing the rows where Neighbourhood = 'Repeating value'

# In[18]:


toronto_data = toronto_data[toronto_data['Neighborhood'] != 'Repeating value']


# In[19]:


print('Result for 1st part : ',toronto_data.shape)


# ## Part-2

# ### Reading geographical coordinates for postal data

# In[20]:


url_pc_toronto = 'http://cocl.us/Geospatial_data'
url_pc_toronto


# In[21]:


pd_postalcode = pd.read_csv(url_pc_toronto)
toronto_pc_data = pd.DataFrame(pd_postalcode)
toronto_pc_data.head()


# ### Merging toronto_data and toronto_pc_data

# In[22]:


toronto_merged = toronto_data
toronto_merged = toronto_merged.join(toronto_pc_data.set_index('Postal Code'), on='Postalcode')
toronto_merged.head(10)


# ### Ignoring the rows where Borough column does not contain 'Toronto' string means keeping only Toronto data

# In[23]:


toronto_filtered = toronto_merged[toronto_merged['Borough'].str.contains('Toronto')]
toronto_filtered.reset_index(drop=True, inplace=True)
toronto_filtered.head()


# In[24]:


toronto_filtered.shape


# ## Part-3

# ### Use geopy library to get the latitude and longitude values of Toronto

# In[25]:


address = 'Toronto, Ontario'

geolocator = Nominatim(user_agent="tor_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('Coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# ### Here, I am going to start utilizing the Foursquare API to explore the neighborhoods and segment them

# In[26]:


CLIENT_ID = 'EU0NRFQMMRG44MUZ5ZECCB44OL3C4HTQXF1QM5EXQVH0KEMB' # your Foursquare ID
CLIENT_SECRET = 'F3CEQR3E3IVCT5JXYF4JKKWC3R4RGL5WLF1XMJ30Z0NWUMM2' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[27]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[28]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[29]:


radius = 500
LIMIT = 100

toronto_venues = getNearbyVenues(names=toronto_filtered['Neighborhood'],
                                   latitudes=toronto_filtered['Latitude'],
                                   longitudes=toronto_filtered['Longitude']
                                  )


# In[30]:


print(toronto_venues.shape)
toronto_venues.head()


# In[31]:


toronto_venues.groupby('Neighborhood').count()


# In[32]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[33]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot.drop(columns=['Neighborhood'], axis=1, inplace=True)
nbhd_list = toronto_venues['Neighborhood']
toronto_onehot.insert(0, 'Neighborhood', nbhd_list)

toronto_onehot.head()


# In[34]:


toronto_onehot.shape


# ### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[35]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[36]:


toronto_grouped.shape


# ### setting a pandas dataframe with highst ten venues for each neighborhood

# In[37]:


#function to sort the venues in descending order

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[38]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### Clustering neighborhoods, running k-means to cluster the neighborhood into 6 clusters

# In[39]:


# set number of clusters
kclusters = 6

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[40]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# In[41]:


toronto_merged = toronto_filtered

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# In[42]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = ['black']
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Now, Examining the Clusters:

# ### Now, you can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, you can then assign a name to each cluster. I will leave this exercise to you.

# #### Cluster 1:
# 

# In[43]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 2:

# In[44]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 3:

# In[45]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 4:

# In[46]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 5:

# In[47]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 6:

# In[48]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 5, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]

