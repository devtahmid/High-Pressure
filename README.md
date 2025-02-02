# HIGH PRESSURE PROJECT


```python
## This is the markdown version of the python notebook which is within this repository folder
```


```python
# Importing necessary libraries for data manipulation and machine learning
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
```


```python
weather_df =  pd.read_csv("weather.txt", 
                      delimiter = ';')
```

#### Abbreviations used in the data:	
**STATIONS ID**	weather station (location)\
**MESS DATUM**	date of measurement, in YYYYMMDD-format\
**QN 3**	unknown, just ignore\
**FX**	probably maximum wind speed\
**FM**	probably average wind speed\
**QN 4**	unknown, just ignore\
**RSK**	total precipitation that day\
**RSKF**	type of precipitation\
**SDK**	sunshine duration in hours\
**SHK TAG**	height of (probably only new?) snow fall, in cm\
**NM**	cloud amount, averaged over the day, from 0.0 to 8.0\
**VPM**	daily mean of vapor pressure in hPa\
**PM**	daily mean of atmospheric pressure in hPa\
**TMK**	daily mean of air temperature in 2m height, in °C\
**UPM**	daily mean of relative humidity, in %\
**TXK**	daily maximum of air temperature in 2m height, in °C\
**TNK**	daily minimum of air temperature in 2m height, in °C\
**TGK**	daily minimum of air temperature in 0m height, in °C\
**eor**	end of row


```python
weather_df # RAW DATA
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STATIONS_ID</th>
      <th>MESS_DATUM</th>
      <th>QN_3</th>
      <th>FX</th>
      <th>FM</th>
      <th>QN_4</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
      <th>eor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1420</td>
      <td>19490101</td>
      <td>-999</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>5</td>
      <td>4.3</td>
      <td>1</td>
      <td>-999.000</td>
      <td>0</td>
      <td>8.0</td>
      <td>6.3</td>
      <td>977.20</td>
      <td>6.6</td>
      <td>65.00</td>
      <td>9.2</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1420</td>
      <td>19490102</td>
      <td>-999</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>5</td>
      <td>2.3</td>
      <td>1</td>
      <td>-999.000</td>
      <td>0</td>
      <td>8.0</td>
      <td>6.9</td>
      <td>981.80</td>
      <td>2.4</td>
      <td>94.00</td>
      <td>7.1</td>
      <td>0.5</td>
      <td>2.2</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1420</td>
      <td>19490103</td>
      <td>-999</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>5</td>
      <td>0.1</td>
      <td>1</td>
      <td>-999.000</td>
      <td>0</td>
      <td>5.0</td>
      <td>6.4</td>
      <td>993.10</td>
      <td>2.0</td>
      <td>88.00</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>-0.3</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1420</td>
      <td>19490104</td>
      <td>-999</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>1</td>
      <td>-999.000</td>
      <td>0</td>
      <td>6.7</td>
      <td>5.6</td>
      <td>1000.40</td>
      <td>2.4</td>
      <td>78.00</td>
      <td>3.7</td>
      <td>-1.6</td>
      <td>-3.4</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1420</td>
      <td>19490105</td>
      <td>-999</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>1</td>
      <td>-999.000</td>
      <td>0</td>
      <td>8.0</td>
      <td>6.7</td>
      <td>1011.20</td>
      <td>3.5</td>
      <td>86.00</td>
      <td>5.3</td>
      <td>1.6</td>
      <td>1.2</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27023</th>
      <td>1420</td>
      <td>20221227</td>
      <td>3</td>
      <td>9.4</td>
      <td>4.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>6</td>
      <td>5.867</td>
      <td>0</td>
      <td>4.1</td>
      <td>6.7</td>
      <td>1014.68</td>
      <td>3.6</td>
      <td>85.17</td>
      <td>6.9</td>
      <td>-1.5</td>
      <td>-3.4</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>27024</th>
      <td>1420</td>
      <td>20221228</td>
      <td>3</td>
      <td>-999.0</td>
      <td>5.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000</td>
      <td>0</td>
      <td>7.9</td>
      <td>7.9</td>
      <td>1004.46</td>
      <td>6.9</td>
      <td>79.29</td>
      <td>10.3</td>
      <td>2.3</td>
      <td>0.6</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>27025</th>
      <td>1420</td>
      <td>20221229</td>
      <td>3</td>
      <td>20.1</td>
      <td>8.2</td>
      <td>3</td>
      <td>4.1</td>
      <td>6</td>
      <td>0.400</td>
      <td>0</td>
      <td>7.8</td>
      <td>9.6</td>
      <td>997.41</td>
      <td>10.8</td>
      <td>74.13</td>
      <td>12.5</td>
      <td>6.3</td>
      <td>2.0</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>27026</th>
      <td>1420</td>
      <td>20221230</td>
      <td>3</td>
      <td>13.4</td>
      <td>4.9</td>
      <td>3</td>
      <td>1.1</td>
      <td>6</td>
      <td>2.117</td>
      <td>0</td>
      <td>6.4</td>
      <td>9.3</td>
      <td>999.98</td>
      <td>8.3</td>
      <td>84.54</td>
      <td>13.4</td>
      <td>4.0</td>
      <td>0.2</td>
      <td>eor</td>
    </tr>
    <tr>
      <th>27027</th>
      <td>1420</td>
      <td>20221231</td>
      <td>3</td>
      <td>17.5</td>
      <td>8.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.833</td>
      <td>0</td>
      <td>7.7</td>
      <td>12.1</td>
      <td>1001.47</td>
      <td>15.5</td>
      <td>68.92</td>
      <td>17.6</td>
      <td>10.9</td>
      <td>5.3</td>
      <td>eor</td>
    </tr>
  </tbody>
</table>
<p>27028 rows × 19 columns</p>
</div>




```python
weather_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STATIONS_ID</th>
      <th>MESS_DATUM</th>
      <th>QN_3</th>
      <th>FX</th>
      <th>FM</th>
      <th>QN_4</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27028.0</td>
      <td>2.702800e+04</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.00000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1420.0</td>
      <td>1.985567e+07</td>
      <td>-257.268462</td>
      <td>-256.581375</td>
      <td>-261.77026</td>
      <td>7.679814</td>
      <td>1.723835</td>
      <td>2.886377</td>
      <td>-16.875645</td>
      <td>0.261211</td>
      <td>5.322384</td>
      <td>9.935674</td>
      <td>1003.152555</td>
      <td>10.311377</td>
      <td>75.139283</td>
      <td>14.707744</td>
      <td>5.815125</td>
      <td>3.943547</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>2.136004e+05</td>
      <td>443.764200</td>
      <td>445.070239</td>
      <td>442.03408</td>
      <td>2.556145</td>
      <td>4.016078</td>
      <td>3.028301</td>
      <td>145.120503</td>
      <td>1.566048</td>
      <td>2.175491</td>
      <td>4.065507</td>
      <td>8.540712</td>
      <td>7.521075</td>
      <td>12.933091</td>
      <td>8.792094</td>
      <td>6.710444</td>
      <td>6.921417</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1420.0</td>
      <td>1.949010e+07</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-999.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>955.900000</td>
      <td>-18.000000</td>
      <td>25.000000</td>
      <td>-15.000000</td>
      <td>-21.600000</td>
      <td>-26.200000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1420.0</td>
      <td>1.967070e+07</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.00000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>6.700000</td>
      <td>998.100000</td>
      <td>4.500000</td>
      <td>66.000000</td>
      <td>7.700000</td>
      <td>0.800000</td>
      <td>-0.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1420.0</td>
      <td>1.985567e+07</td>
      <td>5.000000</td>
      <td>8.200000</td>
      <td>2.30000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.300000</td>
      <td>0.000000</td>
      <td>5.800000</td>
      <td>9.400000</td>
      <td>1003.400000</td>
      <td>10.400000</td>
      <td>77.000000</td>
      <td>14.800000</td>
      <td>5.900000</td>
      <td>4.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1420.0</td>
      <td>2.004070e+07</td>
      <td>10.000000</td>
      <td>11.300000</td>
      <td>3.60000</td>
      <td>10.000000</td>
      <td>1.600000</td>
      <td>6.000000</td>
      <td>7.800000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>12.900000</td>
      <td>1008.600000</td>
      <td>16.200000</td>
      <td>85.000000</td>
      <td>21.600000</td>
      <td>11.125000</td>
      <td>9.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1420.0</td>
      <td>2.022123e+07</td>
      <td>10.000000</td>
      <td>36.000000</td>
      <td>12.80000</td>
      <td>10.000000</td>
      <td>109.700000</td>
      <td>8.000000</td>
      <td>15.900000</td>
      <td>32.000000</td>
      <td>8.000000</td>
      <td>24.900000</td>
      <td>1033.200000</td>
      <td>32.000000</td>
      <td>100.000000</td>
      <td>40.100000</td>
      <td>25.700000</td>
      <td>23.500000</td>
    </tr>
  </tbody>
</table>
</div>



key points:
all stations are the same \
date is in yyyymmdd


```python
weather_df.STATIONS_ID.unique
```




    <bound method Series.unique of 0        1420
    1        1420
    2        1420
    3        1420
    4        1420
             ... 
    27023    1420
    27024    1420
    27025    1420
    27026    1420
    27027    1420
    Name: STATIONS_ID, Length: 27028, dtype: int64>



##### We drop QN_3 and QN_4 because they are irrelevant according to the sheet. STATIONS_ID has the same value as seen in the previous cell so we drop that, along with 'eor' which represents end of line in the raw dataset


```python
weather_df= weather_df.drop(["STATIONS_ID", "QN_3",'QN_4','eor'],axis =1)
```


```python
# Remove all spaces from the name of each column
weather_df.columns = weather_df.columns.str.replace(' ', '')

# Display the updated columns to verify the changes
weather_df.columns
```




    Index(['MESS_DATUM', 'FX', 'FM', 'RSK', 'RSKF', 'SDK', 'SHK_TAG', 'NM', 'VPM',
           'PM', 'TMK', 'UPM', 'TXK', 'TNK', 'TGK'],
          dtype='object')




```python
weather_df= weather_df.replace(-999, np.nan) # replace -999 with null value for ease of operations . -999 are invalid vaues according ot hte sheet
```


```python
weather_df.describe() # final raw data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MESS_DATUM</th>
      <th>FX</th>
      <th>FM</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.702800e+04</td>
      <td>19883.000000</td>
      <td>19881.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>26451.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
      <td>27028.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.985567e+07</td>
      <td>10.208399</td>
      <td>3.255691</td>
      <td>1.723835</td>
      <td>2.886377</td>
      <td>4.548337</td>
      <td>0.261211</td>
      <td>5.322384</td>
      <td>9.935674</td>
      <td>1003.152555</td>
      <td>10.311377</td>
      <td>75.139283</td>
      <td>14.707744</td>
      <td>5.815125</td>
      <td>3.943547</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.136004e+05</td>
      <td>3.776468</td>
      <td>1.600899</td>
      <td>4.016078</td>
      <td>3.028301</td>
      <td>4.308528</td>
      <td>1.566048</td>
      <td>2.175491</td>
      <td>4.065507</td>
      <td>8.540712</td>
      <td>7.521075</td>
      <td>12.933091</td>
      <td>8.792094</td>
      <td>6.710444</td>
      <td>6.921417</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.949010e+07</td>
      <td>1.000000</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>955.900000</td>
      <td>-18.000000</td>
      <td>25.000000</td>
      <td>-15.000000</td>
      <td>-21.600000</td>
      <td>-26.200000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.967070e+07</td>
      <td>7.500000</td>
      <td>2.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>6.700000</td>
      <td>998.100000</td>
      <td>4.500000</td>
      <td>66.000000</td>
      <td>7.700000</td>
      <td>0.800000</td>
      <td>-0.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.985567e+07</td>
      <td>9.800000</td>
      <td>2.900000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.500000</td>
      <td>0.000000</td>
      <td>5.800000</td>
      <td>9.400000</td>
      <td>1003.400000</td>
      <td>10.400000</td>
      <td>77.000000</td>
      <td>14.800000</td>
      <td>5.900000</td>
      <td>4.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.004070e+07</td>
      <td>12.300000</td>
      <td>4.100000</td>
      <td>1.600000</td>
      <td>6.000000</td>
      <td>7.900000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>12.900000</td>
      <td>1008.600000</td>
      <td>16.200000</td>
      <td>85.000000</td>
      <td>21.600000</td>
      <td>11.125000</td>
      <td>9.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.022123e+07</td>
      <td>36.000000</td>
      <td>12.800000</td>
      <td>109.700000</td>
      <td>8.000000</td>
      <td>15.900000</td>
      <td>32.000000</td>
      <td>8.000000</td>
      <td>24.900000</td>
      <td>1033.200000</td>
      <td>32.000000</td>
      <td>100.000000</td>
      <td>40.100000</td>
      <td>25.700000</td>
      <td>23.500000</td>
    </tr>
  </tbody>
</table>
</div>



### We look for the longest stream of rows with no missing values at all and use that subset only


```python
sequence = np.array(weather_df.dropna(how='any').index)                                                                             # source https://stackoverflow.com/questions/54066898/find-longest-subsequence-without-nan-values-in-set-of-series
longest_seq = max(np.split(sequence, np.where(np.diff(sequence) != 1)[0]+1), key=len)    
weather_df= weather_df.iloc[longest_seq]
```


```python
weather_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MESS_DATUM</th>
      <th>FX</th>
      <th>FM</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7305</th>
      <td>19690101</td>
      <td>10.1</td>
      <td>3.9</td>
      <td>2.6</td>
      <td>8</td>
      <td>0.0</td>
      <td>18</td>
      <td>7.3</td>
      <td>5.1</td>
      <td>1015.4</td>
      <td>-1.3</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>-4.9</td>
      <td>-11.5</td>
    </tr>
    <tr>
      <th>7306</th>
      <td>19690102</td>
      <td>8.2</td>
      <td>4.0</td>
      <td>0.3</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>7.7</td>
      <td>6.9</td>
      <td>1010.3</td>
      <td>3.1</td>
      <td>95.0</td>
      <td>4.8</td>
      <td>-0.4</td>
      <td>-1.1</td>
    </tr>
    <tr>
      <th>7307</th>
      <td>19690103</td>
      <td>8.2</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.3</td>
      <td>6.1</td>
      <td>1012.7</td>
      <td>1.8</td>
      <td>85.0</td>
      <td>4.8</td>
      <td>0.5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>7308</th>
      <td>19690104</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>5.6</td>
      <td>1010.0</td>
      <td>0.2</td>
      <td>91.0</td>
      <td>1.6</td>
      <td>-0.3</td>
      <td>-0.8</td>
    </tr>
    <tr>
      <th>7309</th>
      <td>19690105</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.7</td>
      <td>4.8</td>
      <td>1005.0</td>
      <td>-0.6</td>
      <td>80.0</td>
      <td>0.5</td>
      <td>-1.1</td>
      <td>-1.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16204</th>
      <td>19930514</td>
      <td>8.8</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.2</td>
      <td>0</td>
      <td>4.7</td>
      <td>12.7</td>
      <td>988.4</td>
      <td>13.8</td>
      <td>78.0</td>
      <td>18.5</td>
      <td>11.7</td>
      <td>12.7</td>
    </tr>
    <tr>
      <th>16205</th>
      <td>19930515</td>
      <td>19.2</td>
      <td>2.3</td>
      <td>2.1</td>
      <td>6</td>
      <td>4.9</td>
      <td>0</td>
      <td>3.3</td>
      <td>11.6</td>
      <td>996.4</td>
      <td>11.8</td>
      <td>79.0</td>
      <td>19.8</td>
      <td>5.8</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>16206</th>
      <td>19930516</td>
      <td>9.9</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>0</td>
      <td>11.8</td>
      <td>0</td>
      <td>2.7</td>
      <td>9.0</td>
      <td>1006.5</td>
      <td>12.4</td>
      <td>64.0</td>
      <td>18.3</td>
      <td>2.6</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>16207</th>
      <td>19930517</td>
      <td>10.0</td>
      <td>2.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>13.0</td>
      <td>0</td>
      <td>1.7</td>
      <td>10.7</td>
      <td>1001.2</td>
      <td>17.0</td>
      <td>56.0</td>
      <td>24.5</td>
      <td>6.7</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>16208</th>
      <td>19930518</td>
      <td>6.8</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>6</td>
      <td>11.7</td>
      <td>0</td>
      <td>6.3</td>
      <td>13.5</td>
      <td>997.2</td>
      <td>20.8</td>
      <td>55.0</td>
      <td>27.5</td>
      <td>11.9</td>
      <td>9.4</td>
    </tr>
  </tbody>
</table>
<p>8904 rows × 15 columns</p>
</div>




```python
weather_df.shape
```




    (8904, 15)




```python
weather_df.info() # all non-null
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8904 entries, 7305 to 16208
    Data columns (total 15 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   MESS_DATUM  8904 non-null   int64  
     1   FX          8904 non-null   float64
     2   FM          8904 non-null   float64
     3   RSK         8904 non-null   float64
     4   RSKF        8904 non-null   int64  
     5   SDK         8904 non-null   float64
     6   SHK_TAG     8904 non-null   int64  
     7   NM          8904 non-null   float64
     8   VPM         8904 non-null   float64
     9   PM          8904 non-null   float64
     10  TMK         8904 non-null   float64
     11  UPM         8904 non-null   float64
     12  TXK         8904 non-null   float64
     13  TNK         8904 non-null   float64
     14  TGK         8904 non-null   float64
    dtypes: float64(12), int64(3)
    memory usage: 1.1 MB


##### Since weather patterns often align with the month of the years (seasons), the year is removed and only the month and day is kept to be considered


```python
weather_df['MESS_DATUM'] = weather_df['MESS_DATUM'].apply(str).str.slice(4,10)

```


```python
weather_df['MESS_DATUM'] = weather_df['MESS_DATUM'].astype(int) #convert the date to feed into numpy
```


```python
weather_df #final dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MESS_DATUM</th>
      <th>FX</th>
      <th>FM</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7305</th>
      <td>0101</td>
      <td>10.1</td>
      <td>3.9</td>
      <td>2.6</td>
      <td>8</td>
      <td>0.0</td>
      <td>18</td>
      <td>7.3</td>
      <td>5.1</td>
      <td>1015.4</td>
      <td>-1.3</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>-4.9</td>
      <td>-11.5</td>
    </tr>
    <tr>
      <th>7306</th>
      <td>0102</td>
      <td>8.2</td>
      <td>4.0</td>
      <td>0.3</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>7.7</td>
      <td>6.9</td>
      <td>1010.3</td>
      <td>3.1</td>
      <td>95.0</td>
      <td>4.8</td>
      <td>-0.4</td>
      <td>-1.1</td>
    </tr>
    <tr>
      <th>7307</th>
      <td>0103</td>
      <td>8.2</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.3</td>
      <td>6.1</td>
      <td>1012.7</td>
      <td>1.8</td>
      <td>85.0</td>
      <td>4.8</td>
      <td>0.5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>7308</th>
      <td>0104</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>5.6</td>
      <td>1010.0</td>
      <td>0.2</td>
      <td>91.0</td>
      <td>1.6</td>
      <td>-0.3</td>
      <td>-0.8</td>
    </tr>
    <tr>
      <th>7309</th>
      <td>0105</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.7</td>
      <td>4.8</td>
      <td>1005.0</td>
      <td>-0.6</td>
      <td>80.0</td>
      <td>0.5</td>
      <td>-1.1</td>
      <td>-1.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16204</th>
      <td>0514</td>
      <td>8.8</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.2</td>
      <td>0</td>
      <td>4.7</td>
      <td>12.7</td>
      <td>988.4</td>
      <td>13.8</td>
      <td>78.0</td>
      <td>18.5</td>
      <td>11.7</td>
      <td>12.7</td>
    </tr>
    <tr>
      <th>16205</th>
      <td>0515</td>
      <td>19.2</td>
      <td>2.3</td>
      <td>2.1</td>
      <td>6</td>
      <td>4.9</td>
      <td>0</td>
      <td>3.3</td>
      <td>11.6</td>
      <td>996.4</td>
      <td>11.8</td>
      <td>79.0</td>
      <td>19.8</td>
      <td>5.8</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>16206</th>
      <td>0516</td>
      <td>9.9</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>0</td>
      <td>11.8</td>
      <td>0</td>
      <td>2.7</td>
      <td>9.0</td>
      <td>1006.5</td>
      <td>12.4</td>
      <td>64.0</td>
      <td>18.3</td>
      <td>2.6</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>16207</th>
      <td>0517</td>
      <td>10.0</td>
      <td>2.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>13.0</td>
      <td>0</td>
      <td>1.7</td>
      <td>10.7</td>
      <td>1001.2</td>
      <td>17.0</td>
      <td>56.0</td>
      <td>24.5</td>
      <td>6.7</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>16208</th>
      <td>0518</td>
      <td>6.8</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>6</td>
      <td>11.7</td>
      <td>0</td>
      <td>6.3</td>
      <td>13.5</td>
      <td>997.2</td>
      <td>20.8</td>
      <td>55.0</td>
      <td>27.5</td>
      <td>11.9</td>
      <td>9.4</td>
    </tr>
  </tbody>
</table>
<p>8904 rows × 15 columns</p>
</div>



## QR decomposition algorithm and least squares regression by QR


```python
#The Implementation of for QR decomposition Algorithm - Gram  Schmidt Process
def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for k in range(n):
        u = A[:, k]
        for i in range(k):
            R[i, k] = np.dot(Q[:, i].T, A[:, k])
            u = u - R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(u)
        Q[:, k] = u / R[k, k]
    
    return Q, R
```


```python
#The implmentation of Least square regression using QR Decomposition

def least_squares_qr(A, b):
    Q, R = qr_decomposition(A)
    Q_T_y = np.dot( np.linalg.solve(R, QtQ.T, b))
    beta = np.zeros(R.shape[1])
    
    # Back substitution to solve R * beta = Q.T * y
    for i in range(R.shape[1] - 1, -1, -1):
        beta[i] = Q_T_y[i] / R[i, i]
        for j in range(i):
            Q_T_y[j] -= R[j, i] * beta[i]
    
    return beta
```


```python
def least_squares_numpy_qr(A, b):
    """Solve the least squares problem Ax = b using QR decomposition."""
    Q, R = qr_decomposition(A)
    Qt_b = np.dot(Q.T, b)
    x = np.linalg.solve(R, Qt_b)
    return x
```


```python
least_squares_numpy_qr(A,b)
```

#### Implementation of the library (optional)


```python

def least_squares_qr(A, b):
    Q, R = qr_decomposition(A)
    Q_T_b = np.dot(Q.T, b)
    x = np.linalg.solve(R, Q_T_b)
    return x
```

## Implementation


#### Adding target column from current SDK

##### We want to see dependance of various features on the sunshine of the next day. So we make a duplicate of the column and move it one cell upwards


```python
weather_df['target']= weather_df['SDK']

```


```python
weather_df['target'] =weather_df['target'].shift(-1)
weather_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MESS_DATUM</th>
      <th>FX</th>
      <th>FM</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7305</th>
      <td>101</td>
      <td>10.1</td>
      <td>3.9</td>
      <td>2.6</td>
      <td>8</td>
      <td>0.0</td>
      <td>18</td>
      <td>7.3</td>
      <td>5.1</td>
      <td>1015.4</td>
      <td>-1.3</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>-4.9</td>
      <td>-11.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7306</th>
      <td>102</td>
      <td>8.2</td>
      <td>4.0</td>
      <td>0.3</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>7.7</td>
      <td>6.9</td>
      <td>1010.3</td>
      <td>3.1</td>
      <td>95.0</td>
      <td>4.8</td>
      <td>-0.4</td>
      <td>-1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7307</th>
      <td>103</td>
      <td>8.2</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.3</td>
      <td>6.1</td>
      <td>1012.7</td>
      <td>1.8</td>
      <td>85.0</td>
      <td>4.8</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7308</th>
      <td>104</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>5.6</td>
      <td>1010.0</td>
      <td>0.2</td>
      <td>91.0</td>
      <td>1.6</td>
      <td>-0.3</td>
      <td>-0.8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7309</th>
      <td>105</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.7</td>
      <td>4.8</td>
      <td>1005.0</td>
      <td>-0.6</td>
      <td>80.0</td>
      <td>0.5</td>
      <td>-1.1</td>
      <td>-1.8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16202</th>
      <td>512</td>
      <td>10.1</td>
      <td>2.0</td>
      <td>3.7</td>
      <td>6</td>
      <td>7.7</td>
      <td>0</td>
      <td>5.3</td>
      <td>14.8</td>
      <td>992.6</td>
      <td>18.0</td>
      <td>71.0</td>
      <td>24.1</td>
      <td>13.5</td>
      <td>5.3</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>16203</th>
      <td>513</td>
      <td>8.5</td>
      <td>2.6</td>
      <td>2.6</td>
      <td>6</td>
      <td>1.5</td>
      <td>0</td>
      <td>6.7</td>
      <td>15.2</td>
      <td>985.0</td>
      <td>15.4</td>
      <td>85.0</td>
      <td>18.2</td>
      <td>13.0</td>
      <td>12.4</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>16204</th>
      <td>514</td>
      <td>8.8</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.2</td>
      <td>0</td>
      <td>4.7</td>
      <td>12.7</td>
      <td>988.4</td>
      <td>13.8</td>
      <td>78.0</td>
      <td>18.5</td>
      <td>11.7</td>
      <td>12.7</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>16205</th>
      <td>515</td>
      <td>19.2</td>
      <td>2.3</td>
      <td>2.1</td>
      <td>6</td>
      <td>4.9</td>
      <td>0</td>
      <td>3.3</td>
      <td>11.6</td>
      <td>996.4</td>
      <td>11.8</td>
      <td>79.0</td>
      <td>19.8</td>
      <td>5.8</td>
      <td>3.7</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>16206</th>
      <td>516</td>
      <td>9.9</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>0</td>
      <td>11.8</td>
      <td>0</td>
      <td>2.7</td>
      <td>9.0</td>
      <td>1006.5</td>
      <td>12.4</td>
      <td>64.0</td>
      <td>18.3</td>
      <td>2.6</td>
      <td>0.4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8902 rows × 16 columns</p>
</div>




```python
weather_df.drop(weather_df.tail(1).index,inplace=True) # drop last row
weather_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MESS_DATUM</th>
      <th>FX</th>
      <th>FM</th>
      <th>RSK</th>
      <th>RSKF</th>
      <th>SDK</th>
      <th>SHK_TAG</th>
      <th>NM</th>
      <th>VPM</th>
      <th>PM</th>
      <th>TMK</th>
      <th>UPM</th>
      <th>TXK</th>
      <th>TNK</th>
      <th>TGK</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7305</th>
      <td>101</td>
      <td>10.1</td>
      <td>3.9</td>
      <td>2.6</td>
      <td>8</td>
      <td>0.0</td>
      <td>18</td>
      <td>7.3</td>
      <td>5.1</td>
      <td>1015.4</td>
      <td>-1.3</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>-4.9</td>
      <td>-11.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7306</th>
      <td>102</td>
      <td>8.2</td>
      <td>4.0</td>
      <td>0.3</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>7.7</td>
      <td>6.9</td>
      <td>1010.3</td>
      <td>3.1</td>
      <td>95.0</td>
      <td>4.8</td>
      <td>-0.4</td>
      <td>-1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7307</th>
      <td>103</td>
      <td>8.2</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.3</td>
      <td>6.1</td>
      <td>1012.7</td>
      <td>1.8</td>
      <td>85.0</td>
      <td>4.8</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7308</th>
      <td>104</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>5.6</td>
      <td>1010.0</td>
      <td>0.2</td>
      <td>91.0</td>
      <td>1.6</td>
      <td>-0.3</td>
      <td>-0.8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7309</th>
      <td>105</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.7</td>
      <td>4.8</td>
      <td>1005.0</td>
      <td>-0.6</td>
      <td>80.0</td>
      <td>0.5</td>
      <td>-1.1</td>
      <td>-1.8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16201</th>
      <td>511</td>
      <td>8.7</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>0</td>
      <td>11.4</td>
      <td>0</td>
      <td>4.0</td>
      <td>12.2</td>
      <td>998.9</td>
      <td>20.5</td>
      <td>51.0</td>
      <td>27.4</td>
      <td>12.4</td>
      <td>9.5</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>16202</th>
      <td>512</td>
      <td>10.1</td>
      <td>2.0</td>
      <td>3.7</td>
      <td>6</td>
      <td>7.7</td>
      <td>0</td>
      <td>5.3</td>
      <td>14.8</td>
      <td>992.6</td>
      <td>18.0</td>
      <td>71.0</td>
      <td>24.1</td>
      <td>13.5</td>
      <td>5.3</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>16203</th>
      <td>513</td>
      <td>8.5</td>
      <td>2.6</td>
      <td>2.6</td>
      <td>6</td>
      <td>1.5</td>
      <td>0</td>
      <td>6.7</td>
      <td>15.2</td>
      <td>985.0</td>
      <td>15.4</td>
      <td>85.0</td>
      <td>18.2</td>
      <td>13.0</td>
      <td>12.4</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>16204</th>
      <td>514</td>
      <td>8.8</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.2</td>
      <td>0</td>
      <td>4.7</td>
      <td>12.7</td>
      <td>988.4</td>
      <td>13.8</td>
      <td>78.0</td>
      <td>18.5</td>
      <td>11.7</td>
      <td>12.7</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>16205</th>
      <td>515</td>
      <td>19.2</td>
      <td>2.3</td>
      <td>2.1</td>
      <td>6</td>
      <td>4.9</td>
      <td>0</td>
      <td>3.3</td>
      <td>11.6</td>
      <td>996.4</td>
      <td>11.8</td>
      <td>79.0</td>
      <td>19.8</td>
      <td>5.8</td>
      <td>3.7</td>
      <td>11.8</td>
    </tr>
  </tbody>
</table>
<p>8901 rows × 16 columns</p>
</div>



# d) i) Implementation 1: Find relation between Sunshine duration on a day vs. sunshine duration of the following day,


```python
b = weather_df['target'].values
print(b)
```

    [ 0.   0.   0.  ...  3.2  4.9 11.8]



```python
A= weather_df[['SDK']]
print(A)
```

            SDK
    7305    0.0
    7306    0.0
    7307    0.0
    7308    0.0
    7309    0.0
    ...     ...
    16201  11.4
    16202   7.7
    16203   1.5
    16204   3.2
    16205   4.9
    
    [8901 rows x 1 columns]



```python
# Adding a column of ones to X matrices for the intercept term
A = np.hstack((np.ones((A.shape[0], 1)), A))
print(A)
```

    [[1.  0. ]
     [1.  0. ]
     [1.  0. ]
     ...
     [1.  1.5]
     [1.  3.2]
     [1.  4.9]]


### coefficients of the least squared line 


```python
beta = least_squares_qr_numpy(A, b)

```


```python
print(np.shape(A)) 
print(np.shape(b))
```

    (8901, 2)
    (8901,)



```python
y_hat = np.dot(A, beta)
print(y_hat)
```

    [1.80808098 1.80808098 1.80808098 ... 2.68502732 3.67889984 4.67277235]



```python
import matplotlib.pyplot as plt

# Function to plot actual vs. predicted values
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Plot for each model
plot_actual_vs_predicted(b, y_hat, 'Actual vs. Predicted Sunshine Duration (based on all values of previous day)')
```


    
![png](output_43_0.png)
    



```python
# Function to plot actual vs. predicted values with annotations
def plot_with_annotations(y_true, y_pred, start=0, end=100):
    plt.figure(figsize=(20, 8))
    
    # Subset of data for clearer visualization
    y_true_subset = y_true[start:end]
    y_pred_subset= y_pred[start:end]
    days = np.arange(start, end)
    
    # Plot actual values
    plt.plot(days, y_true_subset, label='Actual', color='black', linewidth=2)
    
    # Plot predicted values with smoother lines
    plt.plot(days, y_pred_subset, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Add title and labels
    plt.title('Actual Sunshine vs. Predicted Sunshine ')
    plt.xlabel('Days')
    plt.ylabel('Sunshine duration')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.show()

# Plot the graphs
plot_with_annotations(b, y_hat, start=0, end=365) # change these values to visualise subset of rows

```


    
![png](output_44_0.png)
    



```python
# Calculate R-squared for evaluation
def r_squared(y_target, y_pred):
    ss_total = np.sum((y_target - np.mean(y_target)) ** 2)
    ss_residual = np.sum((y_target - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r_squared_sunshine_onlyQR = r_squared(b, y_hat)

print("R-squared metric:", r_squared_sunshine_onlyQR)

```

    R-squared metric: 0.34171311351224165


R-squared is a statistical measure of how close the data are to the fitted regression line. The definition of R-squared is the percentage of the response variable variation that is explained by a linear model.
~0% r squared => bad
~100% r squared =>  good

# d) ii) Current athmospheric pressure (PM) vs. sunshine duration of the following day.
## Which is the superior predictor?


```python
b = weather_df['target'].values
print(b)
A= weather_df[['PM']]
print(A)
# Adding a column of ones to X matrices for the intercept term
A = np.hstack((np.ones((A.shape[0], 1)), A))
print(A)
### coefficients of the least squared line 
beta = least_squares_qr_numpy(A, b)

print(np.shape(A)) 
print(np.shape(b))
y_hat = np.dot(A, beta)
print(y_hat)
import matplotlib.pyplot as plt

# Function to plot actual vs. predicted values
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Plot for each model
plot_actual_vs_predicted(b, y_hat, 'Actual vs. Predicted Sunshine Duration (based on all values of previous day)')
# Function to plot actual vs. predicted values with annotations
def plot_with_annotations(y_true, y_pred, start=0, end=100):
    plt.figure(figsize=(20, 8))
    
    # Subset of data for clearer visualization
    y_true_subset = y_true[start:end]
    y_pred_subset= y_pred[start:end]
    days = np.arange(start, end)
    
    # Plot actual values
    plt.plot(days, y_true_subset, label='Actual', color='black', linewidth=2)
    
    # Plot predicted values with smoother lines
    plt.plot(days, y_pred_subset, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Add title and labels
    plt.title('Actual Sunshine vs. Predicted Sunshine ')
    plt.xlabel('Days')
    plt.ylabel('Sunshine duration')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.show()

# Plot the graphs
plot_with_annotations(b, y_hat, start=0, end=365) # change these values to visualise subset of rows

```

    [ 0.   0.   0.  ...  3.2  4.9 11.8]
               PM
    7305   1015.4
    7306   1010.3
    7307   1012.7
    7308   1010.0
    7309   1005.0
    ...       ...
    16201   998.9
    16202   992.6
    16203   985.0
    16204   988.4
    16205   996.4
    
    [8901 rows x 1 columns]
    [[1.0000e+00 1.0154e+03]
     [1.0000e+00 1.0103e+03]
     [1.0000e+00 1.0127e+03]
     ...
     [1.0000e+00 9.8500e+02]
     [1.0000e+00 9.8840e+02]
     [1.0000e+00 9.9640e+02]]
    (8901, 2)
    (8901,)
    [5.31237558 4.91671679 5.10290916 ... 2.95393883 3.21771137 3.83835262]



    
![png](output_48_1.png)
    



    
![png](output_48_2.png)
    



```python
# Calculate R-squared for evaluation
def r_squared(y_target, y_pred):
    ss_total = np.sum((y_target - np.mean(y_target)) ** 2)
    ss_residual = np.sum((y_target - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r_squared_sunshine_onlyQR = r_squared(b, y_hat)

print("R-squared metric:", r_squared_sunshine_onlyQR)
```

    R-squared metric: 0.02635105411077443


This is significantly worse compared to using previous day's sunshine, which gave a r-squared value of 0.34171311351224165

At this point using the previous day's sunshine amount is a better predictor than using the atmospheric pressure

### Implementation iii (without svd)

# d) iii) Use all current information to predict the sunshine the next day


```python
b = weather_df['target'].values
A = weather_df.drop(columns=['target']).values

# Adding a column of ones to X matrices for the intercept term
A = np.hstack((np.ones((A.shape[0], 1)), A))


```


```python
A
```




    array([[ 1.00e+00,  1.01e+02,  1.01e+01, ...,  0.00e+00, -4.90e+00,
            -1.15e+01],
           [ 1.00e+00,  1.02e+02,  8.20e+00, ...,  4.80e+00, -4.00e-01,
            -1.10e+00],
           [ 1.00e+00,  1.03e+02,  8.20e+00, ...,  4.80e+00,  5.00e-01,
             1.30e+00],
           ...,
           [ 1.00e+00,  5.13e+02,  8.50e+00, ...,  1.82e+01,  1.30e+01,
             1.24e+01],
           [ 1.00e+00,  5.14e+02,  8.80e+00, ...,  1.85e+01,  1.17e+01,
             1.27e+01],
           [ 1.00e+00,  5.15e+02,  1.92e+01, ...,  1.98e+01,  5.80e+00,
             3.70e+00]])



### coefficients of the least squared line 


```python
beta = least_squares_qr_numpy(A, b)

```


```python
print(np.shape(A)) 
print(np.shape(b))
```

    (8901, 16)
    (8901,)



```python
y_hat = np.dot(A, beta)
print(y_hat)
```

    [1.76685953 2.40341716 2.6760375  ... 3.25115216 4.98900233 5.25648885]



```python
import matplotlib.pyplot as plt

# Function to plot actual vs. predicted values
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Plot for each model
plot_actual_vs_predicted(b, y_hat, 'Actual vs. Predicted Sunshine Duration (based on all values of previous day)')
```


    
![png](output_59_0.png)
    



```python
# Function to plot actual vs. predicted values with annotations
def plot_with_annotations(y_true, y_pred, start=0, end=100):
    plt.figure(figsize=(20, 8))
    
    # Subset of data for clearer visualization
    y_true_subset = y_true[start:end]
    y_pred_subset= y_pred[start:end]
    days = np.arange(start, end)
    
    # Plot actual values
    plt.plot(days, y_true_subset, label='Actual', color='black', linewidth=2)
    
    # Plot predicted values with smoother lines
    plt.plot(days, y_pred_subset, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Add title and labels
    plt.title('Actual vs. Predicted ')
    plt.xlabel('Days')
    plt.ylabel('Sunshine duration')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.show()

# Plot the graphs
plot_with_annotations(b, y_hat, start=0, end=365) # change these values to visualise subset of rows

```


    
![png](output_60_0.png)
    



```python
# Calculate R-squared for evaluation
def r_squared(y_target, y_pred):
    ss_total = np.sum((y_target - np.mean(y_target)) ** 2)
    ss_residual = np.sum((y_target - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r_squared_sunshine_onlyQR = r_squared(b, y_hat)

print("R-squared metric:", r_squared_sunshine_onlyQR)

```

    R-squared metric: 0.4340940727671584


# Prediction of metrics using SVD (Singular Value Decomposition)


```python
b = weather_df['target'].values
A = weather_df.drop(columns=['target']).values
# Adding a column of ones to X matrices for the intercept term
A = np.hstack((np.ones((A.shape[0], 1)), A))

m,n = np.shape(A)
print (m,n)
```

    8901 16



```python
U,S,VT = np.linalg.svd(A)
V = VT.T
```


```python
print("shape of U", np.shape(U), "\n")
print("S", S, "\n")
print("Shape of S", np.shape(S), "\n")
print("Shape of V", np.shape(V), "\n")
```

    shape of U (8901, 8901) 
    
    S [1.15148606e+05 2.68713527e+04 1.59961053e+03 9.85272167e+02
     4.41572094e+02 3.46138330e+02 2.74757935e+02 2.43775647e+02
     2.01197276e+02 1.72128414e+02 1.06459204e+02 9.54615287e+01
     8.15816461e+01 7.78725984e+01 6.03727284e+01 6.88719364e-01] 
    
    Shape of S (16,) 
    
    Shape of V (16, 16) 
    



```python
# append zeroes to the singular matrix 
if m!=n:
    Sigma = np.zeros([m,n])
    for row in range (len(S)):
        Sigma[row,row]=S[row]
else:
    Sigma = np.diag(S)
print(Sigma)
```

    [[115148.60597918      0.              0.         ...      0.
           0.              0.        ]
     [     0.          26871.35270972      0.         ...      0.
           0.              0.        ]
     [     0.              0.           1599.61052571 ...      0.
           0.              0.        ]
     ...
     [     0.              0.              0.         ...      0.
           0.              0.        ]
     [     0.              0.              0.         ...      0.
           0.              0.        ]
     [     0.              0.              0.         ...      0.
           0.              0.        ]]


### Removing columns in U and V to obtain a reduced SVD 
##### Since there are (r number of) rows which are 0 in Sigma, we can remove r number of columns from U. 


```python
# select only first r columns of U and V and first r columns of and rows of Sigma
r= 15
U_r = U[:, 0:r]
Sigma_r = Sigma[0:r, 0:r]  # rxr matrix 
V_r = V[:, 0:r]
A_approximate= np.matmul(np.matmul(U_r, Sigma_r), V_r.T)

print("U_r is: \n", U_r, "\n")
print("Sigma_r is: \n", Sigma_r, "\n")
print("V_r transpose is: \n", V_r.T, "\n")
print("A (rounded) from reduced SVD= \n", np.round(A_approximate, decimals=2), "\n")

print("Original A= \n", A)

frobenius_norm = np.linalg.norm(A - A_approximate, ord='fro')
print("Frobenius norm of A- A_approximate = ", frobenius_norm)
```

    U_r is: 
     [[-0.00770841  0.01902904  0.01794921 ... -0.02893449 -0.01512937
      -0.00606868]
     [-0.00768025  0.01889054  0.01337654 ... -0.02479006  0.00500076
      -0.01476762]
     [-0.00769674  0.01889695  0.00908286 ...  0.00416293  0.00110344
       0.00323904]
     ...
     [-0.00957818  0.0058935  -0.00215124 ...  0.00703689  0.00337786
      -0.00310471]
     [-0.00960302  0.00592636 -0.00460442 ... -0.00641897  0.00982177
       0.00073131]
     [-0.00966517  0.00607506 -0.00162273 ...  0.01942458 -0.03056731
      -0.00907599]] 
    
    Sigma_r is: 
     [[1.15148606e+05 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 2.68713527e+04 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 1.59961053e+03 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 9.85272167e+02
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      4.41572094e+02 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 3.46138330e+02 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 2.74757935e+02 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 2.43775647e+02
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      2.01197276e+02 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 1.72128414e+02 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 1.06459204e+02 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 9.54615287e+01
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      8.15816461e+01 0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 7.78725984e+01 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 6.03727284e+01]] 
    
    V_r transpose is: 
     [[-8.08029590e-04 -5.82064070e-01 -7.81598109e-03 -2.55454619e-03
      -1.36884987e-03 -2.26453553e-03 -3.49750662e-03 -2.44881124e-04
      -4.25526499e-03 -8.02814623e-03 -8.10574485e-01 -8.15091586e-03
      -6.13912120e-02 -1.15825770e-02 -4.57404317e-03 -3.14731096e-03]
     [ 5.80420689e-04 -8.13074782e-01  6.47434812e-03  2.12951180e-03
       6.13732111e-04  2.34246877e-03  3.05443785e-03  1.23838220e-03
       2.95084406e-03  2.09508439e-03  5.80851588e-01  8.13686137e-04
       3.76292760e-02  3.36169095e-03 -2.13834151e-03 -3.42363453e-03]
     [-2.88607864e-05  3.54710125e-03 -3.33731855e-02  1.19925932e-04
       2.19086337e-02  5.30398072e-02 -1.79749610e-01  3.12687069e-02
       5.73943533e-02 -1.62354678e-01 -3.23941534e-02 -3.98410783e-01
       6.04156743e-01 -4.77284132e-01 -3.10458445e-01 -2.89256157e-01]
     [-1.99565826e-04  9.62286411e-03 -4.26536239e-02 -1.24632489e-02
      -1.55319754e-01 -6.67373603e-02  1.66656123e-01  2.13281207e-02
      -1.11026945e-01 -2.44405319e-01  5.82070687e-02 -2.56367586e-01
      -6.97163237e-01 -2.19621497e-01 -3.39525210e-01 -3.87820253e-01]
     [ 6.84815611e-04  2.12208790e-03  7.02687323e-01  2.79630759e-01
       3.52793005e-01  3.19162301e-01 -1.95519967e-01 -1.23171054e-02
       1.43350970e-01 -1.03431637e-01  9.62445500e-03 -1.05899936e-01
      -2.14563027e-01 -2.02577205e-01  6.56409279e-02  1.71424520e-01]
     [ 2.57494853e-04  4.45602263e-04 -7.00881070e-02 -8.72784037e-02
       8.23636384e-01 -1.93257412e-03  2.63464797e-01  2.93880693e-02
      -1.16899129e-01  6.94798942e-02 -8.10045368e-03  8.56130892e-02
       5.61890939e-02  2.39643596e-01 -1.95674540e-01 -3.37711436e-01]
     [ 3.04165789e-04 -2.42853171e-04  5.57525224e-01  1.28355170e-01
      -3.92229691e-01  7.22574583e-02  3.84022847e-01  5.12294374e-02
      -2.17630788e-01  9.38245729e-02 -2.97955272e-02  9.65759520e-02
       2.63714870e-01  3.13304529e-01 -1.81646367e-01 -3.10556900e-01]
     [ 2.16179288e-04  6.10290274e-04 -2.47498823e-01 -1.44286024e-01
      -1.21362217e-01  8.93354929e-01 -8.03003342e-02  1.22238948e-01
       1.32137307e-01  7.90902995e-02  1.77219315e-03  1.23787598e-01
      -5.95458189e-02  1.00438464e-01 -6.38156337e-02 -1.64240855e-01]
     [-3.39044602e-04 -2.78497566e-04 -1.13802807e-01 -5.30841162e-03
       5.65683281e-02  2.52151737e-01  6.07793327e-01  8.04171225e-02
      -4.04311426e-01 -8.35711973e-02 -9.10904048e-04 -3.04568128e-01
       7.23790923e-02 -2.31664942e-01  1.58870539e-01  4.43845422e-01]
     [-4.07739139e-04 -5.40612209e-04 -2.97089365e-02 -1.63208603e-02
      -1.20701537e-02  1.37002419e-01  4.33713922e-02 -9.75483557e-01
      -5.65202997e-02 -1.28638581e-01 -1.15862206e-03 -5.95360127e-03
       3.59410681e-02  5.73984433e-02 -4.56393736e-02 -6.11879113e-03]
     [-6.50843711e-04 -1.82460421e-04  2.63079085e-02 -7.29121064e-02
       1.83156362e-02  5.29520032e-02 -5.46037489e-01  2.31576263e-02
      -7.50732404e-01  1.35715637e-01  7.79554456e-03 -2.38868606e-01
      -4.72044522e-02  2.20189711e-01 -3.70125430e-02  3.06612678e-02]
     [ 8.25481881e-04  2.15441701e-04  4.57402760e-02 -2.24565101e-01
      -7.95269273e-03 -1.44330974e-02 -2.62678220e-02  8.63463802e-02
       2.23601300e-01 -3.63672798e-01 -3.17058214e-03 -2.12667203e-01
       3.57223037e-02  4.59449190e-01 -5.40327047e-01  4.55241214e-01]
     [-1.09357717e-03  1.27944182e-04  2.24140778e-01 -5.78103989e-01
      -1.14128632e-02 -4.51588072e-02  6.73627441e-02 -1.02571199e-01
       1.15767953e-01  6.68394376e-01  4.29641193e-03 -1.06041974e-01
      -9.16967900e-02 -2.35029866e-01 -2.14368982e-01  1.31564794e-01]
     [ 3.13789924e-05 -3.76699531e-05 -2.20334971e-01  6.42141442e-01
      -2.51678094e-05  1.21061391e-02 -6.46607411e-04 -3.75888900e-02
      -4.20419217e-02  3.98181116e-01  9.59700533e-04  1.61488694e-01
      -3.47794937e-02 -1.05236952e-01 -5.30317091e-01  2.40064731e-01]
     [ 5.57140185e-04  1.89871652e-04 -1.03862100e-01  2.72283688e-01
      -1.03002583e-02 -1.20302903e-03  6.04309292e-02 -4.49718318e-02
       2.99806928e-01  3.22095844e-01 -7.23868624e-04 -7.07019431e-01
      -4.28329584e-02  3.66403773e-01  2.46599397e-01 -1.37431543e-01]] 
    
    A (rounded) from reduced SVD= 
     [[ 1.02e+00  1.01e+02  1.01e+01 ... -0.00e+00 -4.90e+00 -1.15e+01]
     [ 1.01e+00  1.02e+02  8.20e+00 ...  4.80e+00 -4.00e-01 -1.10e+00]
     [ 1.01e+00  1.03e+02  8.20e+00 ...  4.80e+00  5.00e-01  1.30e+00]
     ...
     [ 9.80e-01  5.13e+02  8.50e+00 ...  1.82e+01  1.30e+01  1.24e+01]
     [ 9.80e-01  5.14e+02  8.80e+00 ...  1.85e+01  1.17e+01  1.27e+01]
     [ 1.00e+00  5.15e+02  1.92e+01 ...  1.98e+01  5.80e+00  3.70e+00]] 
    
    Original A= 
     [[ 1.00e+00  1.01e+02  1.01e+01 ...  0.00e+00 -4.90e+00 -1.15e+01]
     [ 1.00e+00  1.02e+02  8.20e+00 ...  4.80e+00 -4.00e-01 -1.10e+00]
     [ 1.00e+00  1.03e+02  8.20e+00 ...  4.80e+00  5.00e-01  1.30e+00]
     ...
     [ 1.00e+00  5.13e+02  8.50e+00 ...  1.82e+01  1.30e+01  1.24e+01]
     [ 1.00e+00  5.14e+02  8.80e+00 ...  1.85e+01  1.17e+01  1.27e+01]
     [ 1.00e+00  5.15e+02  1.92e+01 ...  1.98e+01  5.80e+00  3.70e+00]]
    Frobenius norm of A- A_approximate =  0.6887193641600177



```python
frobenius_norm = np.linalg.norm(A, ord='fro')
print("Frobenius norm of A = ", frobenius_norm)
```

    Frobenius norm of A- A_approximate =  118259.70237857869



```python
print("shape of U_r", np.shape(U_r), "\n")
print("Shape of V_r", np.shape(V_r), "\n")
```

    shape of U_r (8901, 15) 
    
    Shape of V_r (16, 15) 
    


#### No information has been lost when we multiply the U_reduced, Sigma_reduced, V_reduced and we get back A. However we dont have some rows and columns of the singular matrix

### Transform the original matrix A
A_reduced ​=U_r.​Σ_r​


```python
A_r = np.matmul(U_r, Sigma_r)
print("Redced A= ", A_r, "\n")
print("Rounded Reduced A= \n", np.round(A_r, decimals =4))
print(np.shape(A_r))
```

    Redced A=  [[-8.87612394e+02  5.11335964e+02  2.87117376e+01 ... -2.36052359e+00
      -1.17816301e+00 -3.66382566e-01]
     [-8.84370321e+02  5.07614307e+02  2.13972577e+01 ... -2.02241351e+00
       3.89422519e-01 -8.91561721e-01]
     [-8.86268878e+02  5.07786650e+02  1.45290359e+01 ...  3.39618871e-01
       8.59273717e-02  1.95549588e-01]
     ...
     [-1.10291462e+03  1.58366399e+02 -3.44114965e+00 ...  5.74080969e-01
       2.63042679e-01 -1.87439787e-01]
     [-1.10577422e+03  1.59249387e+02 -7.36528242e+00 ... -5.23670290e-01
       7.64846894e-01  4.41513080e-02]
     [-1.11293053e+03  1.63244953e+02 -2.59573169e+00 ...  1.58468946e+00
      -2.38035562e+00 -5.47942522e-01]] 
    
    Rounded Reduced A= 
     [[-8.8761240e+02  5.1133600e+02  2.8711700e+01 ... -2.3605000e+00
      -1.1782000e+00 -3.6640000e-01]
     [-8.8437030e+02  5.0761430e+02  2.1397300e+01 ... -2.0224000e+00
       3.8940000e-01 -8.9160000e-01]
     [-8.8626890e+02  5.0778660e+02  1.4529000e+01 ...  3.3960000e-01
       8.5900000e-02  1.9550000e-01]
     ...
     [-1.1029146e+03  1.5836640e+02 -3.4411000e+00 ...  5.7410000e-01
       2.6300000e-01 -1.8740000e-01]
     [-1.1057742e+03  1.5924940e+02 -7.3653000e+00 ... -5.2370000e-01
       7.6480000e-01  4.4200000e-02]
     [-1.1129305e+03  1.6324500e+02 -2.5957000e+00 ...  1.5847000e+00
      -2.3804000e+00 -5.4790000e-01]]
    (8901, 15)



```python
print(np.shape(b))
```

    (8901,)



```python
### coefficients of the least squared line 
beta = least_squares_qr(A_r, b)

```


```python
y_hat = np.dot(A_r, beta)
print(y_hat)
```

    [0.98752049 1.9325047  2.42762189 ... 3.76967521 5.4730664  5.27291426]



```python


# Function to plot actual vs. predicted values
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Plot for each model
plot_actual_vs_predicted(b, y_hat, 'Actual vs. Predicted Sunshine Duration (based on all current information)')
# Function to plot actual vs. predicted values with annotations
def plot_with_annotations(y_true, y_pred, start=0, end=100):
    plt.figure(figsize=(20, 8))
    
    # Subset of data for clearer visualization
    y_true_subset = y_true[start:end]
    y_pred_subset= y_pred[start:end]
    days = np.arange(start, end)
    
    # Plot actual values
    plt.plot(days, y_true_subset, label='Actual', color='black', linewidth=2)
    
    # Plot predicted values with smoother lines
    plt.plot(days, y_pred_subset, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Add title and labels
    plt.title('Actual vs. Predicted ')
    plt.xlabel('Days')
    plt.ylabel('Sunshine duration')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.show()

# Plot the graphs
plot_with_annotations(b, y_hat, start=0, end=365) # change these values to visualise subset of rows


```


    
![png](output_77_0.png)
    



    
![png](output_77_1.png)
    



```python

# Calculate R-squared for evaluation
def r_squared(y_target, y_pred):
    ss_total = np.sum((y_target - np.mean(y_target)) ** 2)
    ss_residual = np.sum((y_target - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r_squared_sunshine_onlyQR = r_squared(b, y_hat)

print("R-squared metric:", r_squared_sunshine_onlyQR)
```

    R-squared metric: 0.4309824502582168


Previous r squared metric without SVD was 0.4340940727671584

After tweaking r in SVD \
r = 15, r squared metric = 0.4309824502582168 \
r = 13, r squared metric = 0.4303715101419606 \
r = 10 , r squared metric = 0.4257325189592154 \
r = 7 , r squared metric = 0.41003188085665954 \
r = 5 , r squared metric = 0.40359803322230203 



## Conlusion 
### When given two metrics Wind pressure and Sunshine value, sunshine value is a better metric to predict the sunshine of the next day. 
### We have used two methods to do the prediction, Least Square Regression method and using SVD (Singular Value Decomposition), and both the methods show the same conclusion. 

