---
title: "Bike Sharing Program: San Francisco & Los Angeles"
date: 2019-01-10
tags: [data science]
---

- The report analyzes two data sets about bike sharing program; one is about Bay Area bike sharing program, and the other one is about Los Angeles (Metro) bike sharing program.

## Bike Sharing in San Francisco
From reading the map of bike sharing station in San Francisco, I found that the bike share stations are centered at the northeast San Francisco because many famous tourist spots, and shopping centers are located in northeast San Francisco. There are many travelers in this area, and tourists may rent a bike to travel around those tourist spots. The mean duration for the bike sharing in San Francisco is 17 minutes, and the mean distance is 1360 meter.
![image](https://drive.google.com/uc?export=view&id=1IdxYOOAfJ-EwrhhKKsGaNHEkOcg6WRAl)

The station which has high frequency as started station indicates that this station is popular. The top5 popular stations are: San Francisco Caltrain (Townsend at 4th), San Francisco Caltrain 2 (330 Townsend), Harry Bridges Plaza (Ferry Building), Embarcadero at Sansome, and 2nd at Townsend.

The San Francisco Caltrain bike sharing station is located near the San Francisco Caltrain station. SF Caltrain is the commuter along the SF Peninsula, through the South Bay to San Jose and Gilroy. It indicates that many Bay Area residents may take the Caltrain to commute instead of driving. For the people who do not drive cars, they may want to rent a bike for convivence, so the SF Caltrain station is one of the most popular bike sharing station. Harry Bridges Plaza station is also a popular bike sharing station because it is near the Pier 33 and Pier 39 which are popular tourist’s spots. With further analyze about Harry Bridges bike share station, I found that as Harry Bridges station as the start station, the most popular end stations are Embarcadero at Sansome, 2nd at Townsend, and San Francisco Caltrain (Townsend at 4th), whereas Embarcadero is also a tourist spot. The mean distance for this station is 1412.554 meters, so it is mostly for short-distance usage.

The bike sharing stations in San Francisco are all located in downtown San Francisco, and the northeast San Francisco. In this area, there are many tourist spots, shopping centers, and office buildings. People may rent a bike for travelling, or for convenient short-distance commuting.

## Bike Sharing in Los Angeles
By seeing the map of bike sharing station in Los Angeles, I found that most bike stations are in the downtown LA, and located in the Pershing Square, which is a public place located in the heart of downtown Los Angeles. Tourist may come to this area, and the residents may also come to Pershing Square for relaxing. The mean duration for LA bike sharing is 3098 meters, and the mean duration for LA bike sharing is 28 minutes.
![image](https://drive.google.com/uc?export=view&id=1--PVjn7ztiwZANxuWNizNnf4UN21vrJj)


The most popular stations are 7th & Flower, Grand & 7th, Broadway & 3rd, Main & 1st, and Union Station West Portal. 7th & Flower is located in front of the 7th street Metro station, located in the Financial District of downtown Los Angeles. 7th street Metro station connects three Metro lines, so there are many people in this station. The mean duration for the 7th & Flower is 23 minutes, and mean distance is 1011 meters, which is much less than the mean distance of the LA bike sharing. It indicates that people who rent bikes from 7th & Flower station is mostly for short-distance commute because people from this station may just want to go to a nearby place in the downtown LA. The other popular bike sharing stations are also located near the heart of downtown LA.

## Usage of Sharing Bike In San Francisco
### Frequency
In the Bay Area, the frequency of the trip is varied in a day. Two peaks of renting bikes is at 8am and 5pm, which are the time of staring of work, and ending of work respectively. During these time periods, people may need to rent a bike to go to work, or go back home, so the frequency of trips are high in these time periods.
![image](https://drive.google.com/uc?export=view&id=1nkajHuFrWl-bDAxTogQjcoorrfXfIrpR)

### Distance
The distance of the trip is varied over hours. From 7pm to 3am next day, the distances of trips are similar to each other. From 4 to 7, the distance of the trips is increasing, and from 7am to 9am, the median distance is the highest, because people may need to ride a bike to work. From 11am to 3pm, the median distance is similar to each other, and started from 3pm, the distance increases until 5pm.
![image](https://drive.google.com/uc?export=view&id=1QOpPdyTi1TKMdvB_8OK8lNn_w0flcavf)

### Duration
During the late night (1am to 3am), the duration of the trips is high because people may want to keep the bikes for a while until the morning. The duration at 4pm is the highest because there might have outliers with high value. From 4pm, the duration starts to increase until 7am. From 7am, the duration keeps constant.
![image](https://drive.google.com/uc?export=view&id=1UJW2gOrUBLZUHJAH2euFU-Ck4s3oLmoY)

### Moving Direction
- morning: 5am – 12pm  afternoon: 12pm – 5pm  evening: 5pm – 9pm  night: 9pm – 5am (next day)
- Bearing is clockwise angle relative to north.

In the morning, people head northwest and southeast the most. In the northwestern of downtown San Francisco, it is a financial district, and office buildings are located in there, so frequency of going to northwestern San Francisco is high. The southeastern San Francisco is Mission district, where many travelers, and artists are there.

In the afternoon, most people head south because there are many restaurants located in the south San Francisco. Tourists may ride bikes from Pier33 to Mission district.

In the evening, the frequency of bearing is diverse because people might go back home, or go for relaxing, so they might ride bikes into different directions.

In the night, most people head south because south San Francisco is the resident’s area, and people are going home in this time period.
![image](https://drive.google.com/uc?export=view&id=1OhEd16XHQuY-bPVuUPmVqoWzChPhXChT)


## Usage of Sharing Bike In Los Angeles
### Frequency
In Los Angeles, from 10pm to 6am in next day, the frequency is low because in that time period, people are rest. From 7am to 21 pm, the frequency has less variation. There is no particularly high frequency in one day because in Los Angeles, people may used bikes for other purposes than just for commuting between work places and house. People may rent bikes to going to public parks, or going shopping in the downtown. The peaks of renting bikes are at 5pm and 6pm.
![image](https://drive.google.com/uc?export=view&id=1tLe-o8UYgQRL2yNmFJJxofrHVJf3b_Ax)

### Distance
The distance of trips has high value at 6am, 7am, 4pm, 5pm and 6pm. Starting from 5am, the distance is increasing; 6am, 7am and 8am are the rush hours because people are going to work or school. 4pm, 5pm, and 6pm are also rush hours, and it is the time period which people are going back home.
![image](https://drive.google.com/uc?export=view&id=1Tp1cS-O3s2zhlYKWekVwFJwZ7w0vwsta)

### Duration
The duration of the trips during the late night (0am, 1am, 2am) has large variation. During the late night, there are extreme high values of duration because people might want to keep the bikes for a long time. People do not ride bikes for commuting, and they might ride bikes just for relaxing, so the duration is long.
![image](https://drive.google.com/uc?export=view&id=1Ve81MPE0eWyCuplRHwY3Je5URwPnsI1p)
