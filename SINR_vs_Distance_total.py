import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import QuantileTransformer 
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


cell = 73
scale = 10

# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)


pattern0 = 'Time : 12:20:04.247'
pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0 = 'Residual Frequency Error :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'cell: {cell} averaged by 10'

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
cellTimes_TM4 = []
cellTimes_TM8 = []
SINR0_TM4 = []
SINR1_TM4 = []
SINR2_TM4 = []
SINR3_TM4 = []
SINR0_TM8 = []
SINR1_TM8 = []
SINR2_TM8 = []
SINR3_TM8 = []

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
                                        cellTimes_TM4.append(time)
                                        split_line0 = line.split(SINR_0)
                                        # Extract the SINR value from the second part of the split line
                                        SINR00 = split_line0[1].strip().replace(' dB',
                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                        SINR0_TM4.append(float(SINR00))
                                        line = next(f)
                                        if SINR_1 in line:
                                            split_line1 = line.split(SINR_1)
                                            # Extract the SINR value from the second part of the split line
                                            SINR11 = split_line1[1].strip().replace(' dB',
                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                            SINR1_TM4.append(float(SINR11))
                                            line = next(f)

                                            if SINR_2 in line:
                                                split_line2 = line.split(SINR_2)
                                                # Extract the SINR value from the second part of the split line
                                                SINR22 = split_line2[1].strip().replace(' dB',
                                                                                            '')  # Remove the 'dB' suffix from the SINR value
                                                SINR2_TM4.append(float(SINR22))
                                                line = next(f)

                                                if SINR_3 in line:
                                                    split_line3 = line.split(SINR_3)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                    SINR3_TM4.append(float(SINR33))
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
                                                cellTimes_TM4.append(time)
                                                split_line0 = line.split(SINR_0)
                                                # Extract the SINR value from the second part of the split line
                                                SINR00 = split_line0[1].strip().replace(' dB',
                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                SINR0_TM4.append(float(SINR00))
                                                line = next(f)
                                                if SINR_1 in line:
                                                    split_line1 = line.split(SINR_1)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                    SINR1_TM4.append(float(SINR11))
                                                    line = next(f)

                                                    if SINR_2 in line:
                                                        split_line2 = line.split(SINR_2)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                    '')  # Remove the 'dB' suffix from the SINR value
                                                        SINR2_TM4.append(float(SINR22))
                                                        line = next(f)

                                                        if SINR_3 in line:
                                                            split_line3 = line.split(SINR_3)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                            SINR3_TM4.append(float(SINR33))
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
                                                        cellTimes_TM4.append(time)
                                                        split_line0 = line.split(SINR_0)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR00 = split_line0[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                        SINR0_TM4.append(float(SINR00))
                                                        line = next(f)
                                                        if SINR_1 in line:
                                                            split_line1 = line.split(SINR_1)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                            SINR1_TM4.append(float(SINR11))
                                                            line = next(f)

                                                            if SINR_2 in line:
                                                                split_line2 = line.split(SINR_2)
                                                                # Extract the SINR value from the second part of the split line
                                                                SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                            '')  # Remove the 'dB' suffix from the SINR value
                                                                SINR2_TM4.append(float(SINR22))
                                                                line = next(f)

                                                                if SINR_3 in line:
                                                                    split_line3 = line.split(SINR_3)
                                                                    # Extract the SINR value from the second part of the split line
                                                                    SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                                    SINR3_TM4.append(float(SINR33))
                                                                    line = next(f)  

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
                                line = next(f)
                                if lockCell in line:
                                    skip_lines(f, 29)
                                    line = next(f)
                                    if SINR_0 in line:
                                        cellTimes_TM8.append(time)
                                        split_line0 = line.split(SINR_0)
                                        # Extract the SINR value from the second part of the split line
                                        SINR00 = split_line0[1].strip().replace(' dB',
                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                        SINR0_TM8.append(float(SINR00))
                                        line = next(f)
                                        if SINR_1 in line:
                                            split_line1 = line.split(SINR_1)
                                            # Extract the SINR value from the second part of the split line
                                            SINR11 = split_line1[1].strip().replace(' dB',
                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                            SINR1_TM8.append(float(SINR11))
                                            line = next(f)

                                            if SINR_2 in line:
                                                split_line2 = line.split(SINR_2)
                                                # Extract the SINR value from the second part of the split line
                                                SINR22 = split_line2[1].strip().replace(' dB',
                                                                                            '')  # Remove the 'dB' suffix from the SINR value
                                                SINR2_TM8.append(float(SINR22))
                                                line = next(f)

                                                if SINR_3 in line:
                                                    split_line3 = line.split(SINR_3)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                    SINR3_TM8.append(float(SINR33))
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
                                                cellTimes_TM8.append(time)
                                                split_line0 = line.split(SINR_0)
                                                # Extract the SINR value from the second part of the split line
                                                SINR00 = split_line0[1].strip().replace(' dB',
                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                SINR0_TM8.append(float(SINR00))
                                                line = next(f)
                                                if SINR_1 in line:
                                                    split_line1 = line.split(SINR_1)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                    SINR1_TM8.append(float(SINR11))
                                                    line = next(f)

                                                    if SINR_2 in line:
                                                        split_line2 = line.split(SINR_2)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                    '')  # Remove the 'dB' suffix from the SINR value
                                                        SINR2_TM8.append(float(SINR22))
                                                        line = next(f)

                                                        if SINR_3 in line:
                                                            split_line3 = line.split(SINR_3)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                            SINR3_TM8.append(float(SINR33))
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
                                                        cellTimes_TM8.append(time)
                                                        split_line0 = line.split(SINR_0)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR00 = split_line0[1].strip().replace(' dB',
                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                        SINR0_TM8.append(float(SINR00))
                                                        line = next(f)
                                                        if SINR_1 in line:
                                                            split_line1 = line.split(SINR_1)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dB' suffix from the SINR value
                                                            SINR1_TM8.append(float(SINR11))
                                                            line = next(f)

                                                            if SINR_2 in line:
                                                                split_line2 = line.split(SINR_2)
                                                                # Extract the SINR value from the second part of the split line
                                                                SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                            '')  # Remove the 'dB' suffix from the SINR value
                                                                SINR2_TM8.append(float(SINR22))
                                                                line = next(f)

                                                                if SINR_3 in line:
                                                                    split_line3 = line.split(SINR_3)
                                                                    # Extract the SINR value from the second part of the split line
                                                                    SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                                '')  # Remove the 'dB' suffix from the SINR value
                                                                    SINR3_TM8.append(float(SINR33))
                                                                    line = next(f)                                       

###############################################################################################################################################################################

# Define constant values for the transformation
const_longitude = 31.34171667
const_latitude = 30.06191389

# Initialize lists to store the transformed coordinates
longitude_list_TM4 = []
latitude_list_TM4 = []
distance_list_TM4 = []
longitude_list_TM8 = []
latitude_list_TM8 = []
distance_list_TM8 = []

with open('newDT/TM4/20230221T121647Z.FMT', 'r') as f:
    for time in cellTimes_TM4:
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
                longitude_list_TM4.append(delta_lon)
                latitude_list_TM4.append(delta_lat)
                distance_list_TM4.append(distance)
                break

with open('newDT/TM8/20230221T105753Z.FMT', 'r') as f:
    for time in cellTimes_TM8:
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
                
                longitude_list_TM8.append(delta_lon)
                latitude_list_TM8.append(delta_lat)
                distance_list_TM8.append(distance)
                break



# Print the lists values

# print('Times:', times)
# print('cellTimes_TM4:', cellTimes_TM4)
# print('cellTimes_TM8:', cellTimes_TM8)
# print('SINR Rx[0] :', SINR0)
# print('SINR Rx[1] :', SINR1)
# print('SINR Rx[2] :', SINR2)
# print('SINR Rx[3] :', SINR3)
# print('Latitudes:', latitude_list)
# print('Longitudes:', longitude_list)
# print('distance_list:', distance_list)
print('Times length:', len(times))
print('cellTimes_TM4 length:', len(cellTimes_TM4))
print('cellTimes_TM8 length:', len(cellTimes_TM8))
print('SINR_TM4 Rx[0] length:', len(SINR0_TM4))
print('SINR_TM4 Rx[1] length:', len(SINR1_TM4))
print('SINR_TM4 Rx[2] length:', len(SINR2_TM4))
print('SINR_TM4 Rx[3] length:', len(SINR3_TM4))
print('SINR_TM8 Rx[0] length:', len(SINR0_TM8))
print('SINR_TM8 Rx[1] length:', len(SINR1_TM8))
print('SINR_TM8 Rx[2] length:', len(SINR2_TM8))
print('SINR_TM8 Rx[3] length:', len(SINR3_TM8))
print('The Final Latitude_TM4 length:', len(latitude_list_TM4))
print('The Final Longitude_TM4 length:', len(longitude_list_TM4))
print('The Final distance_TM4 length:', len(distance_list_TM4))
print('The Final Latitude_TM8 length:', len(latitude_list_TM8))
print('The Final Longitude_TM8 length:', len(longitude_list_TM8))
print('The Final distance_TM8 length:', len(distance_list_TM8))



def calculate_averages(lst):
    averages = []
    for i in range(0, len(lst), scale):
        chunk = lst[i:i+scale]  # Extract a chunk of 5 elements
        avg = sum(chunk) / len(chunk)  # Calculate the average
        averages.append(avg)  # Add the average to the new list
    return averages

def Rx_average(Rx0, Rx1, Rx2, Rx3):
    total = []
    for i in range(0, len(Rx0)):
        avg = ( Rx0[i] + Rx1[i] + Rx2[i] + Rx3[i] ) / 4
        total.append( avg ) # Add the average to the new list
    return total    
def remove_elements(list1, list2, list3, list4, list5):
    i = 0
    while i < len(list1):
        if list1[i] > 1000.0:
            del list1[i]
            del list2[i]
            del list3[i]
            del list4[i]
            del list5[i]
        else:
            i += 1

x0_ave = calculate_averages(distance_list_TM4)
x1_ave = calculate_averages(distance_list_TM8)

y0_ave = calculate_averages(SINR0_TM4)
y1_ave = calculate_averages(SINR1_TM4)
y2_ave = calculate_averages(SINR2_TM4)
y3_ave = calculate_averages(SINR3_TM4)
y00_ave = calculate_averages(SINR0_TM8)
y11_ave = calculate_averages(SINR1_TM8)
y22_ave = calculate_averages(SINR2_TM8)
y33_ave = calculate_averages(SINR3_TM8)

remove_elements(x1_ave, y00_ave, y11_ave, y22_ave, y33_ave)
SINR_TM4_avg = Rx_average(y0_ave, y1_ave, y2_ave, y3_ave)
SINR_TM8_avg = Rx_average(y00_ave, y11_ave, y22_ave, y33_ave)
# int('y00:', len(y00))
print('Avaraged SINR_TM4 length:', len(y0_ave))
print('Avaraged SINR_TM8 length:', len(y00_ave))
print('Averaged Distance_TM4 length:', len(x0_ave))
print('Averaged Distance_TM8 length:', len(x1_ave))
print('Maximum distance TM4 : ', max(x0_ave))
print('Minimum distance TM4 : ', min(x0_ave))
print('Maximum distance TM8 : ', max(x1_ave))
print('Minimum distance TM8 : ', min(x1_ave))
print('Maximum SINR TM4 : ', max(SINR_TM4_avg))
print('Minimum SINR TM4 : ', min(SINR_TM4_avg))
print('Maximum SINR TM8 : ', max(SINR_TM8_avg))
print('Minimum SINR TM8 : ', min(SINR_TM8_avg))


# Create separate arrays for x0, x1, x2, and x3
x0 = np.array(x0_ave).reshape(-1, 1)
x00 = np.array(x1_ave).reshape(-1, 1)

# Create separate arrays for y0, y1, y2, and y3
y0 = np.array(SINR_TM4_avg)
y1 = np.array(SINR_TM8_avg)

scaler0 = QuantileTransformer()
scaler00 = QuantileTransformer()

# Scale the input data
x0_scaled = scaler0.fit_transform(x0)
x00_scaled = scaler00.fit_transform(x00)
print('Maximum Averaged distance TM4 : ', max(x0_scaled))
print('Minimum Averaged distance TM4 : ', min(x0_scaled))
print('Maximum Averaged distance TM8 : ', max(x00_scaled))
print('Minimum Averaged distance TM8 : ', min(x00_scaled))

# Generate the predictions for TM8
x_new0 = np.linspace(x0_scaled.min(), x0_scaled.max(), 100).reshape(-1, 1)

# Generate the predictions for TM4
x_new00 = np.linspace(x00_scaled.min(), x00_scaled.max(), 100).reshape(-1, 1)

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)


# Create a neural network model for TM4
model_tm4 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,), kernel_regularizer=l2(0.01)),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(1)
])



# Compile the TM8 model
model_tm4.compile(optimizer='adam', loss='mean_squared_error')

# Train the TM8 model
history_tm4 = model_tm4.fit(x0_scaled, y0, epochs=10000, verbose=0, validation_split=0.4, callbacks=[early_stopping])

# Create a neural network model for TM4
model_tm8 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,), kernel_regularizer=l2(0.01)),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dense(1)
])


# Compile the TM4 model
model_tm8.compile(optimizer='adam', loss='mean_squared_error')

# Train the TM4 model
history_tm8 = model_tm8.fit(x00_scaled, y1, epochs=10000, verbose=0, validation_split=0.4, callbacks=[early_stopping])


# Predict using the trained model
y0_pred_nn = model_tm4.predict(x_new0)
y1_pred_nn = model_tm8.predict(x_new00)

# Calculate MSE for TM8
mse_tm4 = mean_squared_error(y0, model_tm4.predict(x0_scaled))
print("MSE TM4:", mse_tm4)

# Calculate MSE for TM4
mse_tm8 = mean_squared_error(y1, model_tm8.predict(x00_scaled))
print("MSE TM8:", mse_tm8)

# Plot in the first row, first subplot
fig, axs = plt.subplots()
axs.scatter(x0_scaled, y0, color='b', marker='x', label='Measred TM4')
axs.plot(x_new0, y0_pred_nn, color='yellow', label='Predicted TM4')

axs.set_xlabel('Scaled Distance')
axs.set_ylabel('SINR in db')
axs.set_title('SINR vs Distance')
axs.scatter(x00_scaled, y1, color='r', marker='o', label='Measred TM8')
axs.plot(x_new00, y1_pred_nn, color='darkred', label='Predicted TM8')

axs.legend(loc='upper left')


fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()