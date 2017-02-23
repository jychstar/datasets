dataset is hosted at http://www.bayareabikeshare.com/open-data

Right now it has 3 zip files. Each store a dataset of one year. 

For "Year 1 Data" (2013.8~2014.8,80 MB), it has 2 file folders, each representing half year's data. For 2013.8-2014.2, it has:

1. 201402_status_data.csv :17 million records for bike and dock availability
2. 201402_station_data.csv: 69 stations and their location, dockcount
3. 201402_trip_data.csv: ~144 k instances, 11 features
4. 201402_weather_data.csv – 920 records of daily weather by city

Udacity Data Analyst project "Bay_Area_Bike_Share_Analysis" has remove status_data.csv (622 MB).

Get a small sample,e.g. the first month:

- use csv.reader() and csv.writer() to get objects
- use next() to read from object, use object.writerow() to write into object.

Basic workflow:

- extract from station_data.csv to build a dictionary that maps station_id to the city
- extract from trip_data.csv use `datetime.datetime`module to parse the timestamp strings to get more precise pieces of information
- use `seaborn.countplot(x='weekday', hue="subscription_type", data=trip_data)` to visualize the data