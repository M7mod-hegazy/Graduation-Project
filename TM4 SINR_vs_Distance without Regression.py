import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()



# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)

cell= 383

pattern0 = 'Time : 12:20:04.247'
pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0 = 'Residual Frequency Error :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'TM4_cell: {cell}'

# Define the strings to search for within the block
search_time = 'Time : '
SINR_0 = 'SINR Rx[0] : '
SINR_1 = 'SINR Rx[1] : '
SINR_2 = 'SINR Rx[2] : '
SINR_3 = 'SINR Rx[3] : '
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'


# Initialize empty lists to store the values of interest
times = []
cellTimes = []
SINR0 = []
SINR1 = []
SINR2 = []
SINR3 = []

start_line = 'Time : 12:20:04.247'
with open('newDT/TM4/20230221T121647Z_textexp.txt', 'r') as f:
    # Read the file line by line
    for line in f:
        # Check if the line matches the start of the block pattern
        if pattern1 in line:
            line = next(f)
            if pattern2 in line:
                line = next(f)
                if pattern3 in line:
                    line = next(f)
                    # Read the block until the end and extract the values of interest
                    # Extract the time value if it's found in the line
                    if search_time in line:
                        # Split the line into two parts at the "Time : " string
                        split_line = line.split(search_time)
                        TimeParts = line.split(':')
                        smallTime = TimeParts[3].split('.')
                        if int(TimeParts[1]) >= 12 and int(TimeParts[2]) >= 20 and int(smallTime[0]) >= 4 and int(
                                smallTime[1]) >= 247:
                            # Extract the time value from the second part of the split line
                            time = split_line[1].strip()
                            times.append(time)
                            skip_lines(f, 22)
                            line = next(f)
                            if cell_Patt in line:
                                line = next(f)
                                if lockCell in line:
                                    skip_lines(f, 29)
                                    line = next(f)
                                    if SINR_0 in line:
                                        cellTimes.append(time)
                                        split_line0 = line.split(SINR_0)
                                        # Extract the SINR value from the second part of the split line
                                        SINR00 = split_line0[1].strip().replace(' dB',
                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                        SINR0.append(float(SINR00))
                                        line = next(f)
                                        if SINR_1 in line:
                                            split_line1 = line.split(SINR_1)
                                            # Extract the SINR value from the second part of the split line
                                            SINR11 = split_line1[1].strip().replace(' dB',
                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                            SINR1.append(float(SINR11))
                                            line = next(f)

                                            if SINR_2 in line:
                                                split_line2 = line.split(SINR_2)
                                                # Extract the SINR value from the second part of the split line
                                                SINR22 = split_line2[1].strip().replace(' dB',
                                                                                            '')  # Remove the 'dB' suffix from the SINR value
                                                SINR2.append(float(SINR22))
                                                line = next(f)

                                                if SINR_3 in line:
                                                    split_line3 = line.split(SINR_3)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                    SINR3.append(float(SINR33))
                                                    line = next(f) 
                            
                                else:
                                    skip_lines(f, 39)
                                    line = next(f)
                                    if cell_Patt1 in line:
                                        line = next(f)
                                        if lockCell in line:
                                            skip_lines(f, 29)
                                            line = next(f)
                                            if SINR_0 in line:
                                                cellTimes.append(time)
                                                split_line0 = line.split(SINR_0)
                                                # Extract the SINR value from the second part of the split line
                                                SINR00 = split_line0[1].strip().replace(' dB',
                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                SINR0.append(float(SINR00))
                                                line = next(f)
                                                if SINR_1 in line:
                                                    split_line1 = line.split(SINR_1)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                    SINR1.append(float(SINR11))
                                                    line = next(f)

                                                    if SINR_2 in line:
                                                        split_line2 = line.split(SINR_2)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                    '')  # Remove the 'dB' suffix from the SINR value
                                                        SINR2.append(float(SINR22))
                                                        line = next(f)

                                                        if SINR_3 in line:
                                                            split_line3 = line.split(SINR_3)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                            SINR3.append(float(SINR33))
                                                            line = next(f)
                                        else:                               
                                            skip_lines(f, 39)
                                            line = next(f)
                                            if cell_Patt2 in line:
                                                line = next(f)
                                                if lockCell in line:
                                                    skip_lines(f, 29)
                                                    line = next(f)
                                                    if SINR_0 in line:
                                                        cellTimes.append(time)
                                                        split_line0 = line.split(SINR_0)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR00 = split_line0[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                        SINR0.append(float(SINR00))
                                                        line = next(f)
                                                        if SINR_1 in line:
                                                            split_line1 = line.split(SINR_1)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                            SINR1.append(float(SINR11))
                                                            line = next(f)

                                                            if SINR_2 in line:
                                                                split_line2 = line.split(SINR_2)
                                                                # Extract the SINR value from the second part of the split line
                                                                SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                            '')  # Remove the 'dB' suffix from the SINR value
                                                                SINR2.append(float(SINR22))
                                                                line = next(f)

                                                                if SINR_3 in line:
                                                                    split_line3 = line.split(SINR_3)
                                                                    # Extract the SINR value from the second part of the split line
                                                                    SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                                    SINR3.append(float(SINR33))
                                                                    line = next(f)  


###############################################################################################################################################################################

# Define constant values for the transformation
const_longitude = 31.34171667
const_latitude = 30.06191389

# Initialize lists to store the transformed coordinates
longitude_list = []
latitude_list = []
distance_list = []

with open('newDT/TM4/20230221T121647Z.FMT', 'r') as f:
    for time in cellTimes:
        timePatt = time + "	EQ1		ML1 Serving Cell Measurement Result"
        # Search for the line in the fmt file that contains the time value
        for line in f:
            if timePatt in line:
                # Split the line by tabs to extract the latitude and longitude values
                values = line.split()

                longitude = float(values[9])
                latitude = float(values[8])

                lat1_rad = math.radians(latitude)
                lon1_rad = math.radians(longitude)
                lat2_rad = math.radians(const_latitude)
                lon2_rad = math.radians(const_longitude)

                # Transform the coordinates by subtracting the constant values
                delta_lon  = lat2_rad - lat1_rad
                delta_lat  = lon2_rad - lon1_rad

                a = 6378137  # semi-major axis in meters
                b = 6356752.314245  # semi-minor axis in meters

                haversine = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(
                    delta_lon / 2) ** 2
                c = 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))
                # Compute the distance from the origin using the transformed coordinates
                distance = a * c
                # Add the transformed coordinates and distance to the respective lists
                longitude_list.append(delta_lon)
                latitude_list.append(delta_lat)
                distance_list.append(distance)
                break





# Print the lists values

print('Times:', times)
print('cellTimes:', cellTimes)
print('SINR Rx[0] :', SINR0)
print('SINR Rx[1] :', SINR1)
print('SINR Rx[2] :', SINR2)
print('SINR Rx[3] :', SINR3)
print('Latitudes:', latitude_list)
print('Longitudes:', longitude_list)
print('distance_list:', distance_list)
print('Times length:', len(times))
print('cellTimes length:', len(cellTimes))
print('SINR Rx[0] length:', len(SINR0))
print('SINR Rx[1] length:', len(SINR1))
print('SINR Rx[2] length:', len(SINR2))
print('SINR Rx[3] length:', len(SINR3))
print('The Final Latitude length:', len(latitude_list))
print('The Final Longitude length:', len(longitude_list))
print('The Final distance length:', len(distance_list))





# Reshape the data to match the expected input shape for scikit-learn
X = np.array(distance_list).reshape(-1, 1)
y0 = np.array(SINR0)
y1 = np.array(SINR1)
y2 = np.array(SINR2)
y3 = np.array(SINR3)

# Fit the regression model to the data
regression_model.fit(X, y0)

# Obtain the coefficients and intercept
coefficients = regression_model.coef_
intercept = regression_model.intercept_

print("Coefficient:", coefficients)
print("Intercept:", intercept)

# Make predictions
predicted_values = regression_model.predict(X)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot in the first subplot
axs[0, 0].scatter(distance_list, y0)
axs[0, 0].set_xlabel('Distance in meters')
axs[0, 0].set_ylabel('SINR Rx[0] in dB')
# Plot the regression line

# Fit the regression model to the data for y1
regression_model.fit(X, y1)
predicted_values = regression_model.predict(X)

# Plot in the second subplot
axs[0, 1].scatter(distance_list, y1)
axs[0, 1].set_xlabel('Distance in meters')
axs[0, 1].set_ylabel('SINR Rx[1] in dB')
# Plot the regression line

# Fit the regression model to the data for y2
regression_model.fit(X, y2)
predicted_values = regression_model.predict(X)

# Plot in the third subplot
axs[1, 0].scatter(distance_list, y2)
axs[1, 0].set_xlabel('Distance in meters')
axs[1, 0].set_ylabel('SINR Rx[2] in dB')
# Plot the regression line

# Fit the regression model to the data for y3
regression_model.fit(X, y3)
predicted_values = regression_model.predict(X)

# Plot in the fourth subplot
axs[1, 1].scatter(distance_list, y3)
axs[1, 1].set_xlabel('Distance in meters')
axs[1, 1].set_ylabel('SINR Rx[3] in dB')
# Plot the regression line

# Add the address or title at the top center
fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure with all the plots
plt.show()