import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping




# Define the pattern to match for the start of the block
cell = 373
scale = 10


pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0='Residual Frequency Error :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'Cell: {cell}'
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'


# Define the strings to search for within the block
search_time = 'Time : '
RSRP_0 = 'Inst RSRP Rx[0] : '
RSRP_1 = 'Inst RSRP Rx[1] : '
RSRP_2 = 'Inst RSRP Rx[2] : '
RSRP_3 = 'Inst RSRP Rx[3] : '
ProjSIR_patt = 'Projected SIR :'


# Initialize empty lists to store the values of interest
times = []
cellTimes = []
RSRP0_TM8 = []
RSRP1_TM8 = []
RSRP2_TM8 = []
RSRP3_TM8 = []
RSRP0_TM4 = []
RSRP1_TM4 = []
RSRP2_TM4 = []
RSRP3_TM4 = []
ProjSIR_TM8 = []
ProjSIR_TM4 = []


# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)


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
                                cellTimes.append(time)
                                split_RSRP0 = line.split(RSRP_0)
                                RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                RSRP00 =float(RSRP00) - 30.0 
                                line = next(f)
                                if RSRP_1 in line:
                                    split_RSRP1 = line.split(RSRP_1)
                                    RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                    RSRP11 =float(RSRP11) - 30.0
                                    line = next(f)
                                    if RSRP_2 in line:
                                        split_RSRP2 = line.split(RSRP_2)
                                        RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                        RSRP22 =float(RSRP22) - 30.0
                                        skip_lines(f, 2)
                                        line = next(f)
                                        if RSRP_3 in line:
                                            split_RSRP3 = line.split(RSRP_3)
                                            RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                            RSRP33 =float(RSRP33) - 30.0
                                            skip_lines(f, 18)
                                            line = next(f)
                                            if ProjSIR_patt in line:
                                                split_ProjSIR_TM8 = line.split(ProjSIR_patt)
                                                ProjSIR_TM80 = split_ProjSIR_TM8[1].strip()
                                                if float(ProjSIR_TM80) <= 1000.0:
                                                    RSRP0_TM8.append(float(RSRP00))
                                                    RSRP1_TM8.append(float(RSRP11))
                                                    RSRP2_TM8.append(float(RSRP22))
                                                    RSRP3_TM8.append(float(RSRP33))
                                                    ProjSIR_TM8.append(float(ProjSIR_TM80))
                                                line = next(f)                                                
                                            
                            else:
                                skip_lines(f, 39)
                                line = next(f)
                                if cell_Patt1 in line:
                                    line =next(f)   
                                    if lockCell in line:
                                        skip_lines(f, 9)
                                        line = next(f)
                                        cellTimes.append(time)
                                        split_RSRP0 = line.split(RSRP_0)
                                        RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                        RSRP00 =float(RSRP00) - 30.0 
                                        line = next(f)
                                        if RSRP_1 in line:
                                            split_RSRP1 = line.split(RSRP_1)
                                            RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                            RSRP11 =float(RSRP11) - 30.0
                                            line = next(f)
                                            if RSRP_2 in line:
                                                split_RSRP2 = line.split(RSRP_2)
                                                RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSRP22 =float(RSRP22) - 30.0
                                                skip_lines(f, 2)
                                                line = next(f)
                                                if RSRP_3 in line:
                                                    split_RSRP3 = line.split(RSRP_3)
                                                    RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                    RSRP33 =float(RSRP33) - 30.0
                                                    skip_lines(f, 18)
                                                    line = next(f)
                                                    if ProjSIR_patt in line:
                                                        split_ProjSIR_TM8 = line.split(ProjSIR_patt)
                                                        ProjSIR_TM80 = split_ProjSIR_TM8[1].strip()
                                                        if float(ProjSIR_TM80) <= 1000.0:
                                                            RSRP0_TM8.append(float(RSRP00))
                                                            RSRP1_TM8.append(float(RSRP11))
                                                            RSRP2_TM8.append(float(RSRP22))
                                                            RSRP3_TM8.append(float(RSRP33))
                                                            ProjSIR_TM8.append(float(ProjSIR_TM80))
                                                        line = next(f)   
                                                    
                                    else:  
                                        skip_lines(f, 39)
                                        line = next(f)
                                        if cell_Patt2 in line:
                                            line =next(f)   
                                            if lockCell in line:
                                                skip_lines(f, 9)
                                                line = next(f)
                                                cellTimes.append(time)
                                                split_RSRP0 = line.split(RSRP_0)
                                                RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                RSRP00 =float(RSRP00) - 30.0 
                                                line = next(f)
                                                if RSRP_1 in line:
                                                    split_RSRP1 = line.split(RSRP_1)
                                                    RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                    RSRP11 =float(RSRP11) - 30.0
                                                    line = next(f)
                                                    if RSRP_2 in line:
                                                        split_RSRP2 = line.split(RSRP_2)
                                                        RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSRP22 =float(RSRP22) - 30.0
                                                        skip_lines(f, 2)
                                                        line = next(f)
                                                        if RSRP_3 in line:
                                                            split_RSRP3 = line.split(RSRP_3)
                                                            RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                            RSRP33 =float(RSRP33) - 30.0
                                                            skip_lines(f, 18)
                                                            line = next(f)
                                                            if ProjSIR_patt in line:
                                                                split_ProjSIR_TM8 = line.split(ProjSIR_patt)
                                                                ProjSIR_TM80 = split_ProjSIR_TM8[1].strip()
                                                                if float(ProjSIR_TM80) <= 1000.0:
                                                                    RSRP0_TM8.append(float(RSRP00))
                                                                    RSRP1_TM8.append(float(RSRP11))
                                                                    RSRP2_TM8.append(float(RSRP22))
                                                                    RSRP3_TM8.append(float(RSRP33))
                                                                    ProjSIR_TM8.append(float(ProjSIR_TM80))
                                                                line = next(f)   


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
                                line =next(f)   
                                if lockCell in line:
                                    skip_lines(f, 9)
                                    line = next(f)
                                    cellTimes.append(time)
                                    split_RSRP0 = line.split(RSRP_0)
                                    RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                    RSRP00 =float(RSRP00) - 30.0 
                                    line = next(f)
                                    if RSRP_1 in line:
                                        split_RSRP1 = line.split(RSRP_1)
                                        RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                        RSRP11 =float(RSRP11) - 30.0
                                        line = next(f)
                                        if RSRP_2 in line:
                                            split_RSRP2 = line.split(RSRP_2)
                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                            RSRP22 =float(RSRP22) - 30.0
                                            skip_lines(f, 2)
                                            line = next(f)
                                            if RSRP_3 in line:
                                                split_RSRP3 = line.split(RSRP_3)
                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                RSRP33 =float(RSRP33) - 30.0
                                                skip_lines(f, 18)
                                                line = next(f)
                                                if ProjSIR_patt in line:
                                                    split_ProjSIR_TM4 = line.split(ProjSIR_patt)
                                                    ProjSIR_TM40 = split_ProjSIR_TM4[1].strip()
                                                    if float(ProjSIR_TM40) <= 1000.0:
                                                        RSRP0_TM4.append(float(RSRP00))
                                                        RSRP1_TM4.append(float(RSRP11))
                                                        RSRP2_TM4.append(float(RSRP22))
                                                        RSRP3_TM4.append(float(RSRP33))
                                                        ProjSIR_TM4.append(float(ProjSIR_TM40))
                                                    line = next(f)                                                   
                                                
                                else:
                                    skip_lines(f, 39)
                                    line = next(f)
                                    if cell_Patt1 in line:
                                        line =next(f)   
                                        if lockCell in line:
                                            skip_lines(f, 9)
                                            line = next(f)
                                            cellTimes.append(time)
                                            split_RSRP0 = line.split(RSRP_0)
                                            RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                            RSRP00 =float(RSRP00) - 30.0 
                                            line = next(f)
                                            if RSRP_1 in line:
                                                split_RSRP1 = line.split(RSRP_1)
                                                RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSRP11 =float(RSRP11) - 30.0
                                                line = next(f)
                                                if RSRP_2 in line:
                                                    split_RSRP2 = line.split(RSRP_2)
                                                    RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                    RSRP22 =float(RSRP22) - 30.0
                                                    skip_lines(f, 2)
                                                    line = next(f)
                                                    if RSRP_3 in line:
                                                        split_RSRP3 = line.split(RSRP_3)
                                                        RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                        RSRP33 =float(RSRP33) - 30.0
                                                        skip_lines(f, 18)
                                                        line = next(f)
                                                        if ProjSIR_patt in line:
                                                            split_ProjSIR_TM4 = line.split(ProjSIR_patt)
                                                            ProjSIR_TM40 = split_ProjSIR_TM4[1].strip()
                                                            if float(ProjSIR_TM40) <= 1000.0:
                                                                RSRP0_TM4.append(float(RSRP00))
                                                                RSRP1_TM4.append(float(RSRP11))
                                                                RSRP2_TM4.append(float(RSRP22))
                                                                RSRP3_TM4.append(float(RSRP33))
                                                                ProjSIR_TM4.append(float(ProjSIR_TM40))
                                                            line = next(f)  
                                                        
                                        else:  
                                            skip_lines(f, 39)
                                            line = next(f)
                                            if cell_Patt2 in line:
                                                line =next(f)   
                                                if lockCell in line:
                                                    skip_lines(f, 9)
                                                    line = next(f)
                                                    cellTimes.append(time)
                                                    split_RSRP0 = line.split(RSRP_0)
                                                    RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                    RSRP00 =float(RSRP00) - 30.0 
                                                    line = next(f)
                                                    if RSRP_1 in line:
                                                        split_RSRP1 = line.split(RSRP_1)
                                                        RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSRP11 =float(RSRP11) - 30.0
                                                        line = next(f)
                                                        if RSRP_2 in line:
                                                            split_RSRP2 = line.split(RSRP_2)
                                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                            RSRP22 =float(RSRP22) - 30.0
                                                            skip_lines(f, 2)
                                                            line = next(f)
                                                            if RSRP_3 in line:
                                                                split_RSRP3 = line.split(RSRP_3)
                                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                                RSRP33 =float(RSRP33) - 30.0
                                                                skip_lines(f, 18)
                                                                line = next(f)
                                                                if ProjSIR_patt in line:
                                                                    split_ProjSIR_TM4 = line.split(ProjSIR_patt)
                                                                    ProjSIR_TM40 = split_ProjSIR_TM4[1].strip()
                                                                    if float(ProjSIR_TM40) <= 1000.0:
                                                                        RSRP0_TM4.append(float(RSRP00))
                                                                        RSRP1_TM4.append(float(RSRP11))
                                                                        RSRP2_TM4.append(float(RSRP22))
                                                                        RSRP3_TM4.append(float(RSRP33))
                                                                        ProjSIR_TM4.append(float(ProjSIR_TM40))
                                                                    line = next(f)    
    
                                                                        
###############################################################################################################################################################################


# Print the lists values

# print('Times:', times)
# print('cellTimes:', cellTimes)
# print('ProjSIR_TM8 :', ProjSIR_TM8)
print('Times length:', len(times))
print('ProjSIR_TM8 length:', len(ProjSIR_TM8))
print('ProjSIR_TM4 length:', len(ProjSIR_TM4))
print('RSRP Rx[0] TM4 length:', len(RSRP0_TM4))
print('RSRP Rx[3] TM4 length:', len(RSRP3_TM4))
print('RSRP Rx[0] TM8 length:', len(RSRP0_TM8))
print('RSRP Rx[3] TM8 length:', len(RSRP3_TM8))


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

y0_ave = calculate_averages(ProjSIR_TM8)
y1_ave = calculate_averages(ProjSIR_TM4)

x0_ave = calculate_averages(RSRP0_TM8)
x1_ave = calculate_averages(RSRP1_TM8)
x2_ave = calculate_averages(RSRP2_TM8)
x3_ave = calculate_averages(RSRP3_TM8)
x00_ave = calculate_averages(RSRP0_TM4)
x11_ave = calculate_averages(RSRP1_TM4)
x22_ave = calculate_averages(RSRP2_TM4)
x33_ave = calculate_averages(RSRP3_TM4)

TM8_avg = Rx_average(x0_ave, x1_ave, x2_ave, x3_ave)
TM4_avg = Rx_average(x00_ave, x11_ave, x22_ave, x33_ave)
# int('y00:', len(y00))

print('Maximum Projected SIR TM8 : ', max(y0_ave))
print('Minimum Projected SIR TM8 : ', min(y0_ave))
print('Maximum Projected SIR TM4 : ', max(y1_ave))
print('Minimum Projected SIR TM4 : ', min(y1_ave))
print('Maximum RSRP TM8 : ', max(TM8_avg))
print('Minimum RSRP TM8 : ', min(TM8_avg))
print('Maximum RSRP TM4 : ', max(TM4_avg))
print('Minimum RSRP TM4 : ', min(TM4_avg))


# Create separate arrays for x0, x1, x2, and x3
x0 = np.array(TM8_avg).reshape(-1, 1)
x00 = np.array(TM4_avg).reshape(-1, 1)

# Create separate arrays for y0, y1, y2, and y3
y0 = np.array(y0_ave)
y1 = np.array(y1_ave)

scaler = StandardScaler()

# Scale the input data
x0_scaled = scaler.fit_transform(x0)
x00_scaled = scaler.fit_transform(x00)


# Generate the predictions for TM8
x_new0 = np.linspace(x0_scaled.min(), x0_scaled.max(), 100).reshape(-1, 1)

# Generate the predictions for TM4
x_new00 = np.linspace(x00_scaled.min(), x00_scaled.max(), 100).reshape(-1, 1)

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)


# Create a neural network model for TM8
model_tm8 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])



# Compile the TM8 model
model_tm8.compile(optimizer='adam', loss='mean_squared_error')

# Train the TM8 model
history_tm8 = model_tm8.fit(x0_scaled, y0, epochs=10000, verbose=0, validation_split=0.2, callbacks=[early_stopping])

# Create a neural network model for TM4
model_tm4 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])



# Compile the TM4 model
model_tm4.compile(optimizer='adam', loss='mean_squared_error')

# Train the TM4 model
history_tm4 = model_tm4.fit(x00_scaled, y1, epochs=10000, verbose=0, validation_split=0.2, callbacks=[early_stopping])


# Predict using the trained model
y0_pred_nn = model_tm8.predict(x_new0)
y1_pred_nn = model_tm4.predict(x_new00)

# Calculate MSE for TM8
mse_tm8 = mean_squared_error(y0, model_tm8.predict(x0_scaled))
print("MSE TM8:", mse_tm8)

# Calculate MSE for TM4
mse_tm4 = mean_squared_error(y1, model_tm4.predict(x00_scaled))
print("MSE TM4:", mse_tm4)

# Plot in the first row, first subplot
fig, axs = plt.subplots()
axs.scatter(x0_scaled, y0, color='b', marker='x', label='Measred TM8')
axs.plot(x_new0, y0_pred_nn, color='silver', label='Predicted TM8')
axs.set_ylim(-10, 350)
axs.set_xlabel('Scaled RSRP')
axs.set_ylabel('Projected SIR in db')
axs.set_title('Projected SIR vs. RSRP')
axs.scatter(x00_scaled, y1, color='r', marker='o', label='Measred TM4')
axs.plot(x_new00, y1_pred_nn, color='darkred', label='Predicted TM4')
axs.legend(loc='upper left')


fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

