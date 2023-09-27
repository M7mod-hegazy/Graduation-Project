import matplotlib.pyplot as plt
import re
import math
import numpy as np
from sklearn.linear_model import LinearRegression



# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)

# Define the pattern to match for the start of the block

cell= 73

pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0='Residual Frequency Error :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'TM8_cell: {cell} Averaged by 10'
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'

# Define the strings to search for within the block
search_time = 'Time : '
SINR_0 = 'SINR Rx[0] : '
SINR_1 = 'SINR Rx[1] : '
SINR_2 = 'SINR Rx[2] : '
SINR_3 = 'SINR Rx[3] : '
RSRQ_0 = 'Inst RSRQ Rx[0] : '
RSRQ_1 = 'Inst RSRQ Rx[1] : '
RSRQ_2 = 'Inst RSRQ Rx[2] : '
RSRQ_3 = 'Inst RSRQ Rx[3] : '

# Initialize empty lists to store the values of interest
times = []
cellTimes = []
SINR0 = []
SINR1 = []
SINR2 = []
SINR3 = []
RSRQ0 = []
RSRQ1 = []
RSRQ2 = []
RSRQ3 = []

with open('newDT/TM8/20230221T105753Z_textexp.txt', 'r') as f:
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

                            # Extract the time value from the second part of the split line
                            time = split_line[1].strip()
                            times.append(time)
                            skip_lines(f, 22)
                            line =next(f)
                            if cell_Patt in line:
                                line =next(f)   
                                if lockCell in line:
                                        skip_lines(f, 17)
                                        line = next(f)
                                        split_RSRQ0 = line.split(RSRQ_0)
                                        RSRQ00 = split_RSRQ0[1].strip().replace(' dB','')  # Remove the 'dB' suffix
                                        RSRQ00 =float(RSRQ00)  
                                        RSRQ0.append(float(RSRQ00))
                                        line = next(f)
                                        if RSRQ_1 in line:
                                            split_RSRQ1 = line.split(RSRQ_1)
                                            RSRQ11 = split_RSRQ1[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                            RSRQ11 =float(RSRQ11) 
                                            RSRQ1.append(float(RSRQ11))
                                            line = next(f)
                                            if RSRQ_2 in line:
                                                split_RSRQ2 = line.split(RSRQ_2)
                                                RSRQ22 = split_RSRQ2[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                RSRQ22 =float(RSRQ22) 
                                                RSRQ2.append(float(RSRQ22))
                                                line = next(f)
                                                if RSRQ_3 in line:
                                                    split_RSRQ3 = line.split(RSRQ_3)
                                                    RSRQ33 = split_RSRQ3[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                    RSRQ33 =float(RSRQ33) 
                                                    RSRQ3.append(float(RSRQ33))
                                                    skip_lines(f, 8)
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
                                        line =next(f)   
                                        if lockCell in line:
                                                    skip_lines(f, 17)
                                                    line = next(f)
                                                    split_RSRQ0 = line.split(RSRQ_0)
                                                    RSRQ00 = split_RSRQ0[1].strip().replace(' dB','')  # Remove the 'dB' suffix
                                                    RSRQ00 =float(RSRQ00)  
                                                    RSRQ0.append(float(RSRQ00))
                                                    line = next(f)
                                                    if RSRQ_1 in line:
                                                        split_RSRQ1 = line.split(RSRQ_1)
                                                        RSRQ11 = split_RSRQ1[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                        RSRQ11 =float(RSRQ11) 
                                                        RSRQ1.append(float(RSRQ11))
                                                        line = next(f)
                                                        if RSRQ_2 in line:
                                                            split_RSRQ2 = line.split(RSRQ_2)
                                                            RSRQ22 = split_RSRQ2[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                            RSRQ22 =float(RSRQ22) 
                                                            RSRQ2.append(float(RSRQ22))
                                                            line = next(f)
                                                            if RSRQ_3 in line:
                                                                split_RSRQ3 = line.split(RSRQ_3)
                                                                RSRQ33 = split_RSRQ3[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                                RSRQ33 =float(RSRQ33) 
                                                                RSRQ3.append(float(RSRQ33))
                                                                skip_lines(f, 8)
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
                                                line =next(f)   
                                                if lockCell in line:
                                                            skip_lines(f, 17)
                                                            line = next(f)
                                                            split_RSRQ0 = line.split(RSRQ_0)
                                                            RSRQ00 = split_RSRQ0[1].strip().replace(' dB','')  # Remove the 'dB' suffix
                                                            RSRQ00 =float(RSRQ00)  
                                                            RSRQ0.append(float(RSRQ00))
                                                            line = next(f)
                                                            if RSRQ_1 in line:
                                                                split_RSRQ1 = line.split(RSRQ_1)
                                                                RSRQ11 = split_RSRQ1[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                                RSRQ11 =float(RSRQ11) 
                                                                RSRQ1.append(float(RSRQ11))
                                                                line = next(f)
                                                                if RSRQ_2 in line:
                                                                    split_RSRQ2 = line.split(RSRQ_2)
                                                                    RSRQ22 = split_RSRQ2[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                                    RSRQ22 =float(RSRQ22) 
                                                                    RSRQ2.append(float(RSRQ22))
                                                                    line = next(f)
                                                                    if RSRQ_3 in line:
                                                                        split_RSRQ3 = line.split(RSRQ_3)
                                                                        RSRQ33 = split_RSRQ3[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                                        RSRQ33 =float(RSRQ33) 
                                                                        RSRQ3.append(float(RSRQ33))
                                                                        skip_lines(f, 8)
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

with open('newDT/TM8/20230221T105753Z.FMT', 'r') as f:
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





# print('Times:', times)
print('cellTimes:', cellTimes)
print('SINR Rx[0] :', SINR0[:10])
print('RSRQ Rx[0] :', RSRQ0[:10])
# print('SINR Rx[1] :', SINR1)
# print('SINR Rx[2] :', SINR2)
# print('SINR Rx[3] :', SINR3)
# print('Latitudes:', latitude_list)
# print('Longitudes:', longitude_list)
# print('distance_list:', distance_list)
print('Times length:', len(times))
print('cellTimes length:', len(cellTimes))
print('SINR Rx[0] length:', len(SINR0))
print('SINR Rx[1] length:', len(SINR1))
print('SINR Rx[2] length:', len(SINR2))
print('SINR Rx[3] length:', len(SINR3))
print('RSRQ Rx[0] length:', len(RSRQ0))
print('RSRQ Rx[1] length:', len(RSRQ1))
print('RSRQ Rx[2] length:', len(RSRQ2))
print('RSRQ Rx[3] length:', len(RSRQ3))




def calculate_averages(lst):
    averages = []
    for i in range(0, len(lst), 10):
        chunk = lst[i:i+10]  # Extract a chunk of 5 elements
        avg = sum(chunk) / len(chunk)  # Calculate the average
        averages.append(avg)  # Add the average to the new list
    return averages

y00 = calculate_averages(SINR0)
y11 = calculate_averages(SINR1)
y22 = calculate_averages(SINR2)
y33 = calculate_averages(SINR3)

print('y33:', len(y33))

x00 = calculate_averages(RSRQ0)
x11 = calculate_averages(RSRQ1)
x22 = calculate_averages(RSRQ2)
x33 = calculate_averages(RSRQ3)

y0 = np.array(y00)
y1 = np.array(y11)
y2 = np.array(y22)
y3 = np.array(y33)

x0 = np.array(x00)
x1 = np.array(x11)
x2 = np.array(x22)
x3 = np.array(x33)


# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot in the first subplot
axs[0, 0].scatter(x0, y0, marker='x')
# axs[0, 0].set_xlim(-200, -90)
axs[0, 0].set_xlabel('RSRQ Rx[0] in db')
axs[0, 0].set_ylabel('SINR Rx[0] in db')
axs[0, 0].set_title('SINR Rx[0] vs. RSRQ Rx[0]')


# Plot in the second subplot
axs[0, 1].scatter(x1, y1, marker='x')
# axs[0, 1].set_xlim(-200, -80)
axs[0, 1].set_xlabel('RSRQ Rx[1] in db')
axs[0, 1].set_ylabel('SINR Rx[1] in db')
axs[0, 1].set_title('SINR Rx[1] vs. RSRQ Rx[1]')


# Plot in the second subplot
axs[1, 0].scatter(x2, y2, marker='x')
# axs[1, 0].set_xlim(-200, -90)
axs[1, 0].set_xlabel('RSRQ Rx[2] in db')
axs[1, 0].set_ylabel('SINR Rx[2] in db')
axs[1, 0].set_title('SINR Rx[2] vs. RSRQ Rx[2]')



# Plot in the second subplot
axs[1, 1].scatter(x3, y3, marker='x')
# axs[1, 1].set_xlim(-200, -90)
axs[1, 1].set_xlabel('RSRQ Rx[3] in db')
axs[1, 1].set_ylabel('SINR Rx[3] in db')
axs[1, 1].set_title('SINR Rx[3] vs. RSRQ Rx[3]')


# Add the address or title at the top center
fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure with all the plots
plt.show()