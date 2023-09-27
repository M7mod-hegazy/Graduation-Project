import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


cell = 383
scale = 15

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
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'


# Initialize empty lists to store the values of interest
times = []
cellTimes_TM4 = []
cellTimes_TM8 = []
SINR_0 = 'SINR Rx[0] : '
SINR_1 = 'SINR Rx[1] : '
SINR_2 = 'SINR Rx[2] : '
SINR_3 = 'SINR Rx[3] : '
RSRP_0 = 'Inst RSRP Rx[0] : '
RSRP_1 = 'Inst RSRP Rx[1] : '
RSRP_2 = 'Inst RSRP Rx[2] : '
RSRP_3 = 'Inst RSRP Rx[3] : '

# Initialize empty lists to store the values of interest
SINR0_TM4 = []
SINR1_TM4 = []
SINR2_TM4 = []
SINR3_TM4 = []
SINR0_TM8 = []
SINR1_TM8 = []
SINR2_TM8 = []
SINR3_TM8 = []

RSRP0_TM4 = []
RSRP1_TM4 = []
RSRP2_TM4 = []
RSRP3_TM4 = []
RSRP0_TM8 = []
RSRP1_TM8 = []
RSRP2_TM8 = []
RSRP3_TM8 = []

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
                                    skip_lines(f, 9)
                                    line = next(f)
                                    split_RSRP0 = line.split(RSRP_0)
                                    RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                    RSRP00 =float(RSRP00) - 30.0 
                                    RSRP0_TM4.append(float(RSRP00))
                                    line = next(f)
                                    if RSRP_1 in line:
                                         split_RSRP1 = line.split(RSRP_1)
                                         RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                         RSRP11 =float(RSRP11) - 30.0
                                         RSRP1_TM4.append(float(RSRP11))
                                         line = next(f)
                                         if RSRP_2 in line:
                                            split_RSRP2 = line.split(RSRP_2)
                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                            RSRP22 =float(RSRP22) - 30.0
                                            RSRP2_TM4.append(float(RSRP22))
                                            skip_lines(f, 2)
                                            line = next(f)
                                            if RSRP_3 in line:
                                                split_RSRP3 = line.split(RSRP_3)
                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                RSRP33 =float(RSRP33) - 30.0
                                                RSRP3_TM4.append(float(RSRP33))
                                                line = next(f)
                                               
                            
                                else:
                                    skip_lines(f, 39)
                                    line = next(f)
                                    if cell_Patt1 in line:
                                        line = next(f)
                                        if lockCell in line:
                                            skip_lines(f, 9)
                                            line = next(f)
                                            split_RSRP0 = line.split(RSRP_0)
                                            RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                            RSRP00 =float(RSRP00) - 30.0 
                                            RSRP0_TM4.append(float(RSRP00))
                                            line = next(f)
                                            if RSRP_1 in line:
                                                split_RSRP1 = line.split(RSRP_1)
                                                RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSRP11 =float(RSRP11) - 30.0
                                                RSRP1_TM4.append(float(RSRP11))
                                                line = next(f)
                                                if RSRP_2 in line:
                                                    split_RSRP2 = line.split(RSRP_2)
                                                    RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                    RSRP22 =float(RSRP22) - 30.0
                                                    RSRP2_TM4.append(float(RSRP22))
                                                    skip_lines(f, 2)
                                                    line = next(f)
                                                    if RSRP_3 in line:
                                                        split_RSRP3 = line.split(RSRP_3)
                                                        RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                        RSRP33 =float(RSRP33) - 30.0
                                                        RSRP3_TM4.append(float(RSRP33))
                                                        line = next(f) 
                                        else:                               
                                            skip_lines(f, 39)
                                            line = next(f)
                                            if cell_Patt2 in line:
                                                line = next(f)
                                                if lockCell in line:
                                                    skip_lines(f, 9)
                                                    line = next(f)
                                                    split_RSRP0 = line.split(RSRP_0)
                                                    RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                    RSRP00 =float(RSRP00) - 30.0 
                                                    RSRP0_TM4.append(float(RSRP00))
                                                    line = next(f)
                                                    if RSRP_1 in line:
                                                        split_RSRP1 = line.split(RSRP_1)
                                                        RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSRP11 =float(RSRP11) - 30.0
                                                        RSRP1_TM4.append(float(RSRP11))
                                                        line = next(f)
                                                        if RSRP_2 in line:
                                                            split_RSRP2 = line.split(RSRP_2)
                                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                            RSRP22 =float(RSRP22) - 30.0
                                                            RSRP2_TM4.append(float(RSRP22))
                                                            skip_lines(f, 2)
                                                            line = next(f)
                                                            if RSRP_3 in line:
                                                                split_RSRP3 = line.split(RSRP_3)
                                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                                RSRP33 =float(RSRP33) - 30.0
                                                                RSRP3_TM4.append(float(RSRP33))                
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
                                line =next(f)   
                                if lockCell in line:
                                        skip_lines(f, 9)
                                        line = next(f)
                                        split_RSRP0 = line.split(RSRP_0)
                                        RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                        RSRP00 =float(RSRP00) - 30.0 
                                        RSRP0_TM8.append(float(RSRP00))
                                        line = next(f)
                                        if RSRP_1 in line:
                                            split_RSRP1 = line.split(RSRP_1)
                                            RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                            RSRP11 =float(RSRP11) - 30.0
                                            RSRP1_TM8.append(float(RSRP11))
                                            line = next(f)
                                            if RSRP_2 in line:
                                                split_RSRP2 = line.split(RSRP_2)
                                                RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSRP22 =float(RSRP22) - 30.0
                                                RSRP2_TM8.append(float(RSRP22))
                                                skip_lines(f, 2)
                                                line = next(f)
                                                if RSRP_3 in line:
                                                    split_RSRP3 = line.split(RSRP_3)
                                                    RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                    RSRP33 =float(RSRP33) - 30.0
                                                    RSRP3_TM8.append(float(RSRP33))
                                                    line = next(f)
                                else:
                                    skip_lines(f, 39)
                                    line = next(f)
                                    if cell_Patt1 in line:
                                        line =next(f)   
                                        if lockCell in line:
                                                    skip_lines(f, 9)
                                                    line = next(f)
                                                    split_RSRP0 = line.split(RSRP_0)
                                                    RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                    RSRP00 =float(RSRP00) - 30.0 
                                                    RSRP0_TM8.append(float(RSRP00))
                                                    line = next(f)
                                                    if RSRP_1 in line:
                                                        split_RSRP1 = line.split(RSRP_1)
                                                        RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSRP11 =float(RSRP11) - 30.0
                                                        RSRP1_TM8.append(float(RSRP11))
                                                        line = next(f)
                                                        if RSRP_2 in line:
                                                            split_RSRP2 = line.split(RSRP_2)
                                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                            RSRP22 =float(RSRP22) - 30.0
                                                            RSRP2_TM8.append(float(RSRP22))
                                                            skip_lines(f, 2)
                                                            line = next(f)
                                                            if RSRP_3 in line:
                                                                split_RSRP3 = line.split(RSRP_3)
                                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                                RSRP33 =float(RSRP33) - 30.0
                                                                RSRP3_TM8.append(float(RSRP33))
                                                                line = next(f)
                                        else:  
                                            skip_lines(f, 39)
                                            line = next(f)
                                            if cell_Patt2 in line:
                                                line =next(f)   
                                                if lockCell in line:
                                                            skip_lines(f, 9)
                                                            line = next(f)
                                                            split_RSRP0 = line.split(RSRP_0)
                                                            RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                            RSRP00 =float(RSRP00) - 30.0 
                                                            RSRP0_TM8.append(float(RSRP00))
                                                            line = next(f)
                                                            if RSRP_1 in line:
                                                                split_RSRP1 = line.split(RSRP_1)
                                                                RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                                RSRP11 =float(RSRP11) - 30.0
                                                                RSRP1_TM8.append(float(RSRP11))
                                                                line = next(f)
                                                                if RSRP_2 in line:
                                                                    split_RSRP2 = line.split(RSRP_2)
                                                                    RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                                    RSRP22 =float(RSRP22) - 30.0
                                                                    RSRP2_TM8.append(float(RSRP22))
                                                                    skip_lines(f, 2)
                                                                    line = next(f)
                                                                    if RSRP_3 in line:
                                                                        split_RSRP3 = line.split(RSRP_3)
                                                                        RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                                        RSRP33 =float(RSRP33) - 30.0
                                                                        RSRP3_TM8.append(float(RSRP33))                
                                                                        line = next(f)                                      

###############################################################################################################################################################################



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


x0_ave = calculate_averages(RSRP0_TM4)
x1_ave = calculate_averages(RSRP1_TM4)
x2_ave = calculate_averages(RSRP2_TM4)
x3_ave = calculate_averages(RSRP3_TM4)
x00_ave = calculate_averages(RSRP0_TM8)
x11_ave = calculate_averages(RSRP1_TM8)
x22_ave = calculate_averages(RSRP2_TM8)
x33_ave = calculate_averages(RSRP3_TM8)




RSRP_TM4_avg = Rx_average(x0_ave, x1_ave, x2_ave, x3_ave)
RSRP_TM8_avg = Rx_average(x00_ave, x11_ave, x22_ave, x33_ave)
# int('y00:', len(y00))

print('Avaraged RSRP_TM4 length:', len(x0_ave))
print('Avaraged RSRP_TM8 length:', len(x00_ave))

print('Maximum RSRP TM4 : ', max(RSRP_TM4_avg))
print('Minimum RSRP TM4 : ', min(RSRP_TM4_avg))
print('Maximum RSRP TM8 : ', max(RSRP_TM8_avg))
print('Minimum RSRP TM8 : ', min(RSRP_TM8_avg))


# Create separate arrays for x0, x1, x2, and x3
x0 = np.array(RSRP_TM4_avg).reshape(-1, 1)
x00 = np.array(RSRP_TM8_avg).reshape(-1, 1)

# Create separate arrays for y0, y1, y2, and y3




print('Maximum Averaged RSRP TM4 : ', max(x0))
print('Minimum Averaged RSRP TM4 : ', min(x0))
print('Maximum Averaged RSRP TM8 : ', max(x00))
print('Minimum Averaged RSRP TM8 : ', min(x00))


# Define the ranges for the bars
species = ['Excellent', 'Good', 'Fair', 'Poor']

# Define the start and end numbers for each range
start_numbers = [np.inf, -120, -135, -150]
end_numbers = [-120, -135, -150, -np.inf]

def search_and_select_elements(lst, s, e):
    # Select elements from the start and end of the list
    selected_elements = 0
    for i, num in enumerate(lst):
        if num > e and num < s:
            selected_elements += 1

    return selected_elements

    

range1_TM4 = search_and_select_elements(x0, start_numbers[0], end_numbers[0])
range2_TM4 = search_and_select_elements(x0, start_numbers[1], end_numbers[1])
range3_TM4 = search_and_select_elements(x0, start_numbers[2], end_numbers[2])
range4_TM4 = search_and_select_elements(x0, start_numbers[3], end_numbers[3])

range1_TM8 = search_and_select_elements(x00, start_numbers[0], end_numbers[0])
range2_TM8 = search_and_select_elements(x00, start_numbers[1], end_numbers[1])
range3_TM8 = search_and_select_elements(x00, start_numbers[2], end_numbers[2])
range4_TM8 = search_and_select_elements(x00, start_numbers[3], end_numbers[3])

print('range1_TM4 length:', range1_TM4)
print('range2_TM4 length:', range2_TM4)
print('range3_TM4 length:', range3_TM4)
print('range3_TM4 length:', range3_TM4)
print('range1_TM8 length:', range1_TM8)
print('range2_TM8 length:', range2_TM8)
print('range3_TM8 length:', range3_TM8)
print('range3_TM8 length:', range3_TM8)


penguin_means = {
    'TM4': (range1_TM4, range2_TM4, range3_TM4, range4_TM4),
    'TM8': (range1_TM8, range2_TM8, range3_TM8, range4_TM8),
}

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of data')
ax.set_title(f'RSRP Signal Strength for cell: {cell}')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=2)
# ax.set_ylim(0, 250)

plt.show()