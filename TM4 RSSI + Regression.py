import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping


# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)


cell= 73


pattern0 = 'Time : 12:20:04.247'
pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0 = 'Residual Frequency Error :'
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'TM4_cell: {cell} average by 10'

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

print('y33:', len(y33))

x00 = calculate_averages(RSSI0)
x11 = calculate_averages(RSSI1)
x22 = calculate_averages(RSSI2)
x33 = calculate_averages(RSSI3)

# Create separate arrays for x0, x1, x2, and x3
x0 = np.array(x00).reshape(-1, 1)
x1 = np.array(x11).reshape(-1, 1)
x2 = np.array(x22).reshape(-1, 1)
x3 = np.array(x33).reshape(-1, 1)

# Create separate arrays for y0, y1, y2, and y3
y0 = np.array(y00)
y1 = np.array(y11)
y2 = np.array(y22)
y3 = np.array(y33)
# Scale the features
scaler = StandardScaler()

# Scale the input data
x0_scaled = scaler.fit_transform(x0)
x1_scaled = scaler.fit_transform(x1)
x2_scaled = scaler.fit_transform(x2)
x3_scaled = scaler.fit_transform(x3)

# Generate the predictions for TM4
x_new0 = np.linspace(x0_scaled.min(), x0_scaled.max(), 100).reshape(-1, 1)
x_new1 = np.linspace(x1_scaled.min(), x1_scaled.max(), 100).reshape(-1, 1)
x_new2 = np.linspace(x2_scaled.min(), x2_scaled.max(), 100).reshape(-1, 1)
x_new3 = np.linspace(x3_scaled.min(), x3_scaled.max(), 100).reshape(-1, 1)


early_stopping = EarlyStopping(patience=10, restore_best_weights=True)


# Create a neural network model for TM4
model0 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model1 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model2 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model3 = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])


# Compile the TM4 model
model0.compile(optimizer='adam', loss='mean_squared_error')
model1.compile(optimizer='adam', loss='mean_squared_error')
model2.compile(optimizer='adam', loss='mean_squared_error')
model3.compile(optimizer='adam', loss='mean_squared_error')

# Train the TM4 model
history_0 = model0.fit(x0_scaled, y0, epochs=5000, verbose=0, validation_split=0.3, callbacks=[early_stopping])
history_1 = model1.fit(x1_scaled, y1, epochs=5000, verbose=0, validation_split=0.3, callbacks=[early_stopping])
history_2 = model2.fit(x2_scaled, y2, epochs=5000, verbose=0, validation_split=0.3, callbacks=[early_stopping])
history_3 = model3.fit(x3_scaled, y3, epochs=5000, verbose=0, validation_split=0.3, callbacks=[early_stopping])


# Predict using the trained model
y0_pred_nn = model0.predict(x_new0)
y1_pred_nn = model1.predict(x_new1)
y2_pred_nn = model2.predict(x_new2)
y3_pred_nn = model3.predict(x_new3)

# Calculate MSE for TM4
mse_0 = mean_squared_error(y0, model0.predict(x0_scaled))
mse_1 = mean_squared_error(y1, model1.predict(x1_scaled))
mse_2 = mean_squared_error(y2, model2.predict(x2_scaled))
mse_3 = mean_squared_error(y3, model3.predict(x3_scaled))
print("MSE0:", mse_0)
print("MSE1:", mse_1)
print("MSE2:", mse_2)
print("MSE3:", mse_3)


# Create a figure and axes objects
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot in the first subplot
axs[0, 0].scatter(x0_scaled, y0, color='b', marker='x', label='Measred')
axs[0, 0].plot(x_new0, y0_pred_nn, color='r', label='Predicted')
# axs[0, 0].set_xlim(-3.5, 2.5)
axs[0, 0].set_xlabel('Scaled RSSI Rx[0]')
axs[0, 0].set_ylabel('SINR Rx[0] in db')
axs[0, 0].set_title('SINR Rx[0] vs. RSSI Rx[0]')
axs[0, 0].legend(loc='lower right')

# Plot in the second subplot
axs[0, 1].scatter(x1_scaled, y1, color='b', marker='x', label='Measred')
axs[0, 1].plot(x_new1, y1_pred_nn, color='r', label='Predicted')
# axs[0, 1].set_xlim(-3.5, 2.5)
axs[0, 1].set_xlabel('Scaled RSSI Rx[1]')
axs[0, 1].set_ylabel('SINR Rx[1] in db')
axs[0, 1].set_title('SINR Rx[1] vs. RSSI Rx[1]')
axs[0, 1].legend(loc='lower right')

# Plot in the third subplot
axs[1, 0].scatter(x2_scaled, y2, color='b', marker='x', label='Measred')
axs[1, 0].plot(x_new2, y2_pred_nn, color='r', label='Predicted')
# axs[1, 0].set_xlim(-3.5, 2.5)
axs[1, 0].set_xlabel('Scaled RSSI Rx[2]')
axs[1, 0].set_ylabel('SINR Rx[2] in db')
axs[1, 0].set_title('SINR Rx[2] vs. RSSI Rx[2]')
axs[1, 0].legend(loc='lower right')

# Plot in the fourth subplot
axs[1, 1].scatter(x3_scaled, y3, color='b', marker='x', label='Measred')
axs[1, 1].plot(x_new3, y3_pred_nn, color='r', label='Predicted')
# axs[1, 1].set_xlim(-3.5, 2.5)
axs[1, 1].set_xlabel('Scaled RSSI Rx[3]')
axs[1, 1].set_ylabel('SINR Rx[3] in db')
axs[1, 1].set_title('SINR Rx[3] vs. RSSI Rx[3]')
axs[1, 1].legend(loc='lower right')

fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()
