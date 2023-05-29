#!/usr/bin/env python

"""
orcaMain.py

Project: Where's the Whale, and Which Whale is It?
For Stanford University CS109 Spring 2023

Description: this file accomplishes four principal goals.
1. Given a date, where (in the Salish Sea) is the most likely place to encounter a
   Southern Resident Killer Whale (SRKW)?
2. Given a location and a date, what is the actual probability of encountering a
   SRKW?
3. Given location, date, and that I have seen a SRKW, which pod (J, K, or L) is the
   most likely to be seen?
4. Using an exponential random variable, how long would we expect to wait until the
   next sighting of a SRKW?

This project uses a combination of statistical analyses, including bootstrapping,
exponential random variable models, and 'area inflation' to find results. Data is from
The Whale Museum's Orca Master dataset in WA State.

Author: Nathanael James Cadicamo

Date: 26 May 2023

"""

# libraries to import
import csv
import random
import numpy as np
from datetime import datetime
from scipy import stats

# defining CSV data sets for later access; stored locally on natecadicamo macbook
DATA_total = '/Users/natecadicamo/Desktop/orcas/Cadicamo_SRKW_DataRequest.csv'
DATA_clean = '/Users/natecadicamo/Desktop/orcas/orcaDataClean.csv'


# Part 1: get date from user, find most likely location [long, lat] to encounter SRKW

def find_location(month):
    """
    Input: user date MMDD that they want to search for SRKW
    Output: most likely coordinate location to encounter SRKW
    """
    # make list of all sighting locations
    sightings = []

    # fill in sightings according to month
    with open(DATA_clean, 'r') as f:
        reader = csv.reader(f)
        # skip header row
        next(reader)
        # iterate over all the relevant data
        for row in reader:
            sighting_date = row[0]
            sighting_date.split('/')
            sighting_month = int(sighting_date[0])
            if month == sighting_month:
                sightings.append((float(row[2]), float(row[3])))

    # error case: no data for that month
    if len(sightings) == 0:
        print("No data on orcas for that month. Try a warmer time of year!")
        return ['error', -1]

    # find and print best location
    print("Calculating... this will take a minute...")
    best = best_location(sightings)
    print(f'The best location is {best}.')
    print('(Note that even though this location has the highest probability, orcas are still quite rare!)')

    # return loc to other functions
    return [best, sightings]


def best_location(sightings):
    """
    Input: sightings, list of [(latitude, longitude)] SRKW sightings
    Output: [lat, long] single location
    Approach: count sightings in a grid, then find best grid prob
    """
    # initialize an unnecessarily large grid
    grid = [[0 for _ in range(20000)] for _ in range(20000)]

    # fill into grid, rounding to approximately 1 km
    for s in sightings:
        x = round(s[0] * 100)
        y = round(s[1] * -100)
        grid[x][y] += 1

    # find best location
    best = (0, 0)
    prob = 0
    for row in range(20000):
        for col in range(20000):
            if grid[row][col] > prob:
                prob = grid[row][col]
                best = (row / 100, - col / 100)
    return best


# Part 2: given date and location, find probability of encountering SRKW

def find_prob(loc, sightings):
    """
    Input: location (lat, long) given the month; sightings [(lat, long)]
    Output: actual probability of seeing a SRKW at location
    Approach: area bootstrap
    """
    # find relevant bounds: 111 km approximate daily range of SRKW
    lat_max = loc[0] + 1
    lat_min = loc[0] - 1
    long_max = loc[1] + 1
    long_min = loc[1] - 1
    whale_bounds = [lat_max, lat_min, long_max, long_min]

    # get all sightings in the nearby bounds
    nearby_points = []
    for s in sightings:
        if lat_min < s[0] < lat_max:
            if long_min < s[1] < long_max:
                nearby_points.append(s)

    # inflate points to be areas in sighting range 0.01, approximately 1 km
    nearby_areas = []
    for p in nearby_points:
        x_max = p[0] + 0.01
        x_min = p[0] - 0.01
        y_max = p[1] + 0.01
        y_min = p[1] - 0.01
        nearby_areas.append([x_max, x_min, y_max, y_min])

    # bounds for visible sighting from loc
    sight_x_max = loc[0] + 0.01
    sight_x_min = loc[0] - 0.01
    sight_y_max = loc[1] + 0.01
    sight_y_min = loc[1] - 0.01
    sight_bounds = [sight_x_max, sight_x_min, sight_y_max, sight_y_min]

    # call bootstrap function
    prob = bootstrap(whale_bounds, sight_bounds, nearby_areas)
    print(f'The probability of seeing a SRKW here is {round(prob, 6)}.')
    if prob == 0:
        print('They are rare and elusive creatures... someday, somewhere –– but not today, and not here!')
    return prob


def bootstrap(whale_bounds, sight_bounds, nearby_areas):
    """
    Input: whale_bounds, total range of interest; sight_bounds, total visible range;
    nearby_areas, list of ranges of orcas
    Output: probability of random sample from whale_bounds being in sight_bounds
    """
    # generate sample space as area of whale bounds
    sample_space = (whale_bounds[0] - whale_bounds[1]) * (whale_bounds[2] - whale_bounds[3])
    # print(sample_space)

    # find event space as total area of nearby areas
    event_space = 0
    for a in nearby_areas:
        event_space += (a[0] - a[1]) * (a[2] - a[3])
    # print(event_space)

    # basic bernoulli probability of orca event in sample space
    p = event_space / sample_space

    # now, bootstrap it with discrete events
    orcas_in_sight = 0
    large_number = 100000
    for i in range(large_number):
        # random chance
        chance = random.random()
        # success case
        if chance <= p:
            # random generation of orca coordinates in given bounds
            orca_lat = random.uniform(whale_bounds[1], whale_bounds[0])
            orca_long = random.uniform(whale_bounds[3], whale_bounds[2])
            # check if these coordinates are within visible bounds
            if sight_bounds[1] < orca_lat < sight_bounds[0] and sight_bounds[3] < orca_long < sight_bounds[2]:
                orcas_in_sight += 1

    # return the bootstrapped probability
    return orcas_in_sight / large_number


# Part 3: given date, location, and encounter, find most likely pod (J, K, or L)

def most_likely_pod(month, loc):
    """
    Input: month, as int; loc, as [float x, float y]
    Output: pod (J, K, or L) and P(pod)
    """
    # find loc bounds: 111 km approximate daily range of SRKW
    lat_max = loc[0] + 1
    lat_min = loc[0] - 1
    long_max = loc[1] + 1
    long_min = loc[1] - 1

    # relevant sightings
    pod_sightings = []

    # fill in sightings according to month, loc area
    with open(DATA_clean, 'r') as f:
        reader = csv.reader(f)
        # skip header row
        next(reader)
        # iterate over all the relevant data
        for row in reader:
            sighting_date = row[0]
            sighting_date.split('/')
            sighting_month = int(sighting_date[0])
            if month == sighting_month:
                x = float(row[2])
                y = float(row[3])
                # check if sighting is within bounds
                if lat_min < x < lat_max:
                    if long_min < y < long_max:
                        pod_sightings.append(row[1])

    # pod counts with Laplace smoothing
    pod_counts = {'J': 1, 'K': 1, 'L': 1}
    total = 3

    # get probabilities
    total += len(pod_sightings)
    for s in pod_sightings:
        if 'J' in s:
            pod_counts['J'] += 1
        if 'K' in s:
            pod_counts['K'] += 1
        if 'L' in s:
            pod_counts['L'] += 1

    # print the most likely pod
    max_pod = max(pod_counts, key=pod_counts.get)
    prob = pod_counts[max_pod] / total
    print(f'\nIf you do see a SRKW, the most likely pod is {max_pod} pod, which has a probability of {round(prob, 3)}.')

    # print the probabilities for each pod
    for key in pod_counts:
        pod_counts[key] /= total
        pod_counts[key] = round(pod_counts[key], 3)
    print(f'In case you were wondering, the probabilities for each are: {pod_counts}.')
    print('(Note that you may see more than 1 pod...)\n')


# Part 4: exponential random variable model of time until next sighting

def time_until_next():
    """
    Output: expectation of exponential RV, time until the next SRKW sighting
    """
    # initialize an empty list of timestamps
    timestamps = []

    # get dates from total dataset
    with open(DATA_total, 'r') as f:
        reader = csv.reader(f)
        # skip header row
        next(reader)
        # get each date
        for row in reader:
            date_string = row[0]
            # convert string to datetime object, then timestamp
            date_object = datetime.strptime(date_string, "%m/%d/%y")
            # convert the timestamp from seconds to hours
            timestamp = date_object.timestamp() / 3600
            # only add if given date has not been considered
            if timestamp not in timestamps:
                timestamps.append(timestamp)

    # calculate time differences between consecutive events
    time_diffs = np.diff(timestamps)

    # calculate average time difference, which gives expected value
    avg_time_diff = np.mean(time_diffs)
    print(f'The expected time to wait for the next SRKW sighting is {round(avg_time_diff, 3)} hours.')

    # set random exponential value with scale as 1 / lambda
    exp = stats.expon(scale=avg_time_diff)

    # user interaction
    print('Enter time to wait for the next SRKW in hours, and we will give the probability that it takes that long.')
    stop = False
    while not stop:
        time = input('Enter waiting time (hours): ')
        print(f'You would wait {time} or more hours with probability {round(1 - exp.cdf(float(time)), 3)}.\n')
        keep_going = input('Want to try another time? Y/N: ')
        if keep_going == 'N' or keep_going == 'n':
            stop = True


# main() and user interaction below

def get_user_date():
    date = input("What date are you looking for a Southern Resident Killer Whale? MMDD: ")

    # ensure it is valid
    try:
        month = int(date[0:2])
    except ValueError:
        # if invalid, recurse
        print("Invalid date format. Try again please.")
        get_user_date()
        return

    return month


def main():
    # get the date from user
    date = get_user_date()

    # GOAL 1: find most likely location
    find_loc = find_location(date)
    location = find_loc[0]
    sightings = find_loc[1]
    if location == 'error':
        main()
        return

    # we can either use the calculated best location, or any other
    use_loc = input('\nDo you want to use this location? Y/N: ')
    if use_loc == 'N' or use_loc == 'n':
        print('Okay, enter any other coordinates in the Salish Sea.')
        lat = float(input('Latitude: '))
        long = float(input('Longitude: '))
        location = [lat, long]

    # GOAL 2: find probability of SRKW sighting
    find_prob(location, sightings)

    # GOAL 3: find most likely pod
    most_likely_pod(date, location)

    # GOAL 4: estimate time to wait and associated probabilities
    time_until_next()

    # finish up
    print('\nNice to chat with you. Hope you find the whale you are looking for!')


if __name__ == '__main__':
    main()
