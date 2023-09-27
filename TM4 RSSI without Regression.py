import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

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
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'TM4_cell: {cell} Averaged by 10'

# Define the strings to search for within the block
search_time = 'Time : '
SINR_0 = 'SINR Rx[0] : '
SINR_1 = 'SINR Rx[1] : '
SINR_2 = 'SINR Rx[2] : '
SINR_3 = 'SINR Rx[3] : '
RSSI_0 = 'Inst RSSI Rx[0] : '
RSSI_1 = 'Inst RSSI Rx[1] : '
RSSI_2 = 'Inst RSSI Rx[2] : '
RSSI_3 = 'Inst RSSI Rx[3] : '

# Initialize empty lists to store the values of interest
times = []
cellTimes = []
SINR0 = []
SINR1 = []
SINR2 = []
SINR3 = []
RSSI0 = []
RSSI1 = []
RSSI2 = []
RSSI3 = []

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
                                    skip_lines(f, 23)
                                    line = next(f)
                                    split_RSSI0 = line.split(RSSI_0)
                                    RSSI00 = split_RSSI0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                    RSSI00 =float(RSSI00) - 30.0  
                                    RSSI0.append(float(RSSI00))
                                    line = next(f)
                                    if RSSI_1 in line:
                                         split_RSSI1 = line.split(RSSI_1)
                                         RSSI11 = split_RSSI1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                         RSSI11 =float(RSSI11) - 30.0 
                                         RSSI1.append(float(RSSI11))
                                         line = next(f)
                                         if RSSI_2 in line:
                                            split_RSSI2 = line.split(RSSI_2)
                                            RSSI22 = split_RSSI2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                            RSSI22 =float(RSSI22) - 30.0
                                            RSSI2.append(float(RSSI22))
                                            line = next(f)
                                            if RSSI_3 in line:
                                                split_RSSI3 = line.split(RSSI_3)
                                                RSSI33 = split_RSSI3[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSSI33 =float(RSSI33) - 30.0 
                                                RSSI3.append(float(RSSI33))
                                                skip_lines(f, 2)
                                                line = next(f)
                                                if SINR_0 in line:
                                                    cellTimes.append(time)
                                                    split_line0 = line.split(SINR_0)
                                                    # Extract the SINR value from the second part of the split line
                                                    SINR00 = split_line0[1].strip().replace(' dB',
                                                                                            '')  # Remove the 'dBm' suffix from the SINR value
                                                    SINR0.append(float(SINR00))
                                                    line = next(f)
                                                    if SINR_1 in line:
                                                        split_line1 = line.split(SINR_1)
                                                        # Extract the SINR value from the second part of the split line
                                                        SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                    '')  # Remove the 'dBm' suffix from the SINR value
                                                        SINR1.append(float(SINR11))
                                                        line = next(f)

                                                        if SINR_2 in line:
                                                            split_line2 = line.split(SINR_2)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                        '')  # Remove the 'dBm' suffix from the SINR value
                                                            SINR2.append(float(SINR22))
                                                            line = next(f)

                                                            if SINR_3 in line:
                                                                split_line3 = line.split(SINR_3)
                                                                # Extract the SINR value from the second part of the split line
                                                                SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                            '')  # Remove the 'dBm' suffix from the SINR value
                                                                SINR3.append(float(SINR33))
                                                                line = next(f) 
                            
                                else:
                                    skip_lines(f, 39)
                                    line = next(f)
                                    if cell_Patt1 in line:
                                        line = next(f)
                                        if lockCell in line:
                                            skip_lines(f, 23)
                                            line = next(f)
                                            split_RSSI0 = line.split(RSSI_0)
                                            RSSI00 = split_RSSI0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                            RSSI00 =float(RSSI00) - 30.0  
                                            RSSI0.append(float(RSSI00))
                                            line = next(f)
                                            if RSSI_1 in line:
                                                split_RSSI1 = line.split(RSSI_1)
                                                RSSI11 = split_RSSI1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSSI11 =float(RSSI11) - 30.0 
                                                RSSI1.append(float(RSSI11))
                                                line = next(f)
                                                if RSSI_2 in line:
                                                    split_RSSI2 = line.split(RSSI_2)
                                                    RSSI22 = split_RSSI2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                    RSSI22 =float(RSSI22) - 30.0 
                                                    RSSI2.append(float(RSSI22))
                                                    line = next(f)
                                                    if RSSI_3 in line:
                                                        split_RSSI3 = line.split(RSSI_3)
                                                        RSSI33 = split_RSSI3[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSSI33 =float(RSSI33) - 30.0 
                                                        RSSI3.append(float(RSSI33))
                                                        skip_lines(f, 2)
                                                        line = next(f)
                                                        if SINR_0 in line:
                                                            cellTimes.append(time)
                                                            split_line0 = line.split(SINR_0)
                                                            # Extract the SINR value from the second part of the split line
                                                            SINR00 = split_line0[1].strip().replace(' dB',
                                                                                                    '')  # Remove the 'dBm' suffix from the SINR value
                                                            SINR0.append(float(SINR00))
                                                            line = next(f)
                                                            if SINR_1 in line:
                                                                split_line1 = line.split(SINR_1)
                                                                # Extract the SINR value from the second part of the split line
                                                                SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                            '')  # Remove the 'dBm' suffix from the SINR value
                                                                SINR1.append(float(SINR11))
                                                                line = next(f)

                                                                if SINR_2 in line:
                                                                    split_line2 = line.split(SINR_2)
                                                                    # Extract the SINR value from the second part of the split line
                                                                    SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                                '')  # Remove the 'dBm' suffix from the SINR value
                                                                    SINR2.append(float(SINR22))
                                                                    line = next(f)

                                                                    if SINR_3 in line:
                                                                        split_line3 = line.split(SINR_3)
                                                                        # Extract the SINR value from the second part of the split line
                                                                        SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                                    '')  # Remove the 'dBm' suffix from the SINR value
                                                                        SINR3.append(float(SINR33))
                                                                        line = next(f) 
                                        else:                               
                                            skip_lines(f, 39)
                                            line = next(f)
                                            if cell_Patt2 in line:
                                                line = next(f)
                                                if lockCell in line:
                                                    skip_lines(f, 23)
                                                    line = next(f)
                                                    split_RSSI0 = line.split(RSSI_0)
                                                    RSSI00 = split_RSSI0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                    RSSI00 =float(RSSI00) - 30.0  
                                                    RSSI0.append(float(RSSI00))
                                                    line = next(f)
                                                    if RSSI_1 in line:
                                                        split_RSSI1 = line.split(RSSI_1)
                                                        RSSI11 = split_RSSI1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSSI11 =float(RSSI11) - 30.0 
                                                        RSSI1.append(float(RSSI11))
                                                        line = next(f)
                                                        if RSSI_2 in line:
                                                            split_RSSI2 = line.split(RSSI_2)
                                                            RSSI22 = split_RSSI2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                            RSSI22 =float(RSSI22) - 30.0
                                                            RSSI2.append(float(RSSI22))
                                                            line = next(f)
                                                            if RSSI_3 in line:
                                                                split_RSSI3 = line.split(RSSI_3)
                                                                RSSI33 = split_RSSI3[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                                RSSI33 =float(RSSI33) - 30.0 
                                                                RSSI3.append(float(RSSI33))
                                                                skip_lines(f, 2)
                                                                line = next(f)
                                                                if SINR_0 in line:
                                                                    cellTimes.append(time)
                                                                    split_line0 = line.split(SINR_0)
                                                                    # Extract the SINR value from the second part of the split line
                                                                    SINR00 = split_line0[1].strip().replace(' dB',
                                                                                                            '')  # Remove the 'dBm' suffix from the SINR value
                                                                    SINR0.append(float(SINR00))
                                                                    line = next(f)
                                                                    if SINR_1 in line:
                                                                        split_line1 = line.split(SINR_1)
                                                                        # Extract the SINR value from the second part of the split line
                                                                        SINR11 = split_line1[1].strip().replace(' dB',
                                                                                                                    '')  # Remove the 'dBm' suffix from the SINR value
                                                                        SINR1.append(float(SINR11))
                                                                        line = next(f)

                                                                        if SINR_2 in line:
                                                                            split_line2 = line.split(SINR_2)
                                                                            # Extract the SINR value from the second part of the split line
                                                                            SINR22 = split_line2[1].strip().replace(' dB',
                                                                                                                        '')  # Remove the 'dBm' suffix from the SINR value
                                                                            SINR2.append(float(SINR22))
                                                                            line = next(f)

                                                                            if SINR_3 in line:
                                                                                split_line3 = line.split(SINR_3)
                                                                                # Extract the SINR value from the second part of the split line
                                                                                SINR33 = split_line3[1].strip().replace(' dB',
                                                                                                                            '')  # Remove the 'dBm' suffix from the SINR value
                                                                                SINR3.append(float(SINR33))
                                                                                line = next(f)  

          
# Print the lists values

print('Times:', times[-10:])
print('cellTimes:', cellTimes[-10:])
print('SINR Rx[0] :', SINR0[-10:])
print('RSSI Rx[0] :', RSSI0[-10:])
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
print('RSSI Rx[0] length:', len(RSSI0))
print('RSSI Rx[1] length:', len(RSSI1))
print('RSSI Rx[2] length:', len(RSSI2))
print('RSSI Rx[3] length:', len(RSSI3))


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

x00 = calculate_averages(RSSI0)
x11 = calculate_averages(RSSI1)
x22 = calculate_averages(RSSI2)
x33 = calculate_averages(RSSI3)

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
axs[0, 0].set_xlabel('RSSI Rx[0] in dB')
axs[0, 0].set_ylabel('SINR Rx[0] in dB')
axs[0, 0].set_title('SINR Rx[0] vs. RSSI Rx[0]')


# Plot in the second subplot
axs[0, 1].scatter(x1, y1, marker='x')
# axs[0, 1].set_xlim(-200, -80)
axs[0, 1].set_xlabel('RSSI Rx[1] in dB')
axs[0, 1].set_ylabel('SINR Rx[1] in dB')
axs[0, 1].set_title('SINR Rx[1] vs. RSSI Rx[1]')


# Plot in the second subplot
axs[1, 0].scatter(x2, y2, marker='x')
# axs[1, 0].set_xlim(-200, -90)
axs[1, 0].set_xlabel('RSSI Rx[2] in dB')
axs[1, 0].set_ylabel('SINR Rx[2] in dB')
axs[1, 0].set_title('SINR Rx[2] vs. RSSI Rx[2]')



# Plot in the second subplot
axs[1, 1].scatter(x3, y3, marker='x')
# axs[1, 1].set_xlim(-200, -90)
axs[1, 1].set_xlabel('RSSI Rx[3] in dB')
axs[1, 1].set_ylabel('SINR Rx[3] in dB')
axs[1, 1].set_title('SINR Rx[3] vs. RSSI Rx[3]')


# Add the address or title at the top center
fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure with all the plots
plt.show()
