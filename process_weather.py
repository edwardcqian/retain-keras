import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from os import listdir
from os.path import isfile, join

from datetime import datetime, timedelta

# used for location indexing
locations = ['DATETIME'  , 'VANCOUVER'  , 'PORTLAND'    , 'SAN FRANCISCO', 'SEATTLE', 
            'LOS ANGELES', 'SAN DIEGO'  , 'LAS VEGAS'   , 'PHOENIX'      , 'ALBUQUERQUE', 
            'DENVER'     , 'SAN ANTONIO', 'DALLAS'      , 'HOUSTON'      , 'KANSAS CITY', 
            'MINNEAPOLIS', 'SAINT LOUIS', 'CHICAGO'     , 'NASHVILLE'    , 'INDIANAPOLIS', 
            'ATLANTA'    , 'DETROIT'    , 'JACKSONVILLE', 'CHARLOTTE'    , 'MIAMI', 
            'PITTSBURGH' , 'TORONTO'    , 'PHILADELPHIA', 'NEW YORK'     , 'MONTREAL', 
            'BOSTON'     , 'BEERSHEBA'  , 'TEL AVIV DISTRICT', 'EILAT', 'HAIFA', 'NAHARIYYA', 'JERUSALEM']

loc_dic = {}
for loc in locations:
    loc_dic[loc] = len(loc_dic)

print("loading extreme weather data")
path_ex = '/home/edwardqian/Documents/weather/extreme_weather.csv'


print("loading hourly weather data")
df_dic = {}

mypath = '/home/edwardqian/Documents/weather/hourly/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
    name = f.split('.')[0]
    temp = pd.read_csv(mypath+f)
    temp['datetime'] = temp['datetime'].apply(lambda x : datetime.strptime(x,"%Y-%m-%d %H:%M:%S")).unique()
    df_dic[name] = temp



ex_weather = pd.read_csv(path_ex)

ex_weather['BEGIN_DATE_TIME'] = ex_weather['BEGIN_DATE_TIME'].apply(lambda x : datetime.strptime(x,"%d-%b-%y %H:%M:%S"))

ex_weather = ex_weather.sort_values(by='BEGIN_DATE_TIME')

ex_weather = ex_weather[ex_weather['BEGIN_DATE_TIME']>df_dic['humidity']['datetime'].iloc[0]]

ex_weather = ex_weather.drop_duplicates(subset='EVENT_ID')

all_df = pd.DataFrame()

disc_dic = {}

for i,row in ex_weather.iterrows():
    temp_df = pd.DataFrame()
    date = row['BEGIN_DATE_TIME']
    location = row['LOCATION'].title()
    for df in df_dic:
        cur_df = df_dic[df]
        result = cur_df[(cur_df['datetime']>date-timedelta(days=7))&(cur_df['datetime']<date+timedelta(days=1))][location]
        temp_df[df] = result
    temp_df['datetime'] = cur_df[(cur_df['datetime']>date-timedelta(days=7))&(cur_df['datetime']<date+timedelta(days=1))]['datetime']
    codes = []
    numerics = []
    for j,day in temp_df.iterrows():
        code = day['weather_description']
        numeric = list(day[['wind_speed','temperature','pressure','humidity','wind_direction']].values)
        if code in disc_dic:
            pass
        else:
            disc_dic[code] = len(disc_dic)
        code = disc_dic[code]
        codes.append([code])
        numerics.append(numeric)
    new_df = pd.DataFrame()
    new_df['codes'] = [codes]
    new_df['numerics'] = [numerics]
    new_df['index'] = row['EVENT_ID']
    all_df = all_df.append(new_df)

all_targets = ex_weather[['EVENT_ID','EVENT_TYPE']]

all_targets.columns = ['index','target']

all_targets['target'] = ((all_targets['target']=='Flood')|(all_targets['target']=='Flash Flood')).astype('int')

data_train,data_test,target_train,target_test = train_test_split(all_df, all_targets, test_size=0.1, random_state=12345)

out_directory = '.'

data_train.sort_index().to_pickle(out_directory+'/data_train_weather.pkl')
data_test.sort_index().to_pickle(out_directory+'/data_test_weather.pkl')
target_train.sort_index().to_pickle(out_directory+'/target_train_weather.pkl')
target_test.sort_index().to_pickle(out_directory+'/target_test_weather.pkl')
pickle.dump(disc_dic, open(out_directory+'/dictionary_weather.pkl', 'wb'), -1)