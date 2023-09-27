import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

regression_model = LinearRegression()

# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)

cell= 73
poly_degree = 4
alpha=0.01
lambda_=0.01
iterations=10000

pattern0 = 'Time : 12:20:04.247'
pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0 = 'Residual Frequency Error :'
lockCell = f'Physical Cell ID : {cell}'
pltAddress = f'TM4_cell: {cell}'
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'

# Define the strings to search for within the block
search_time = 'Time : '
SINR_0 = 'SINR Rx[0] : '
SINR_1 = 'SINR Rx[1] : '
SINR_2 = 'SINR Rx[2] : '
SINR_3 = 'SINR Rx[3] : '
RSRP_0 = 'Inst RSRP Rx[0] : '
RSRP_1 = 'Inst RSRP Rx[1] : '
RSRP_2 = 'Inst RSRP Rx[2] : '
RSRP_3 = 'Inst RSRP Rx[3] : '

# Initialize empty lists to store the values of interest
times = []
cellTimes = []
SINR0 = []
SINR1 = []
SINR2 = []
SINR3 = []
RSRP0 = []
RSRP1 = []
RSRP2 = []
RSRP3 = []


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
                                    RSRP0.append(float(RSRP00))
                                    line = next(f)
                                    if RSRP_1 in line:
                                         split_RSRP1 = line.split(RSRP_1)
                                         RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                         RSRP11 =float(RSRP11) - 30.0
                                         RSRP1.append(float(RSRP11))
                                         line = next(f)
                                         if RSRP_2 in line:
                                            split_RSRP2 = line.split(RSRP_2)
                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                            RSRP22 =float(RSRP22) - 30.0
                                            RSRP2.append(float(RSRP22))
                                            skip_lines(f, 2)
                                            line = next(f)
                                            if RSRP_3 in line:
                                                split_RSRP3 = line.split(RSRP_3)
                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                RSRP33 =float(RSRP33) - 30.0
                                                RSRP3.append(float(RSRP33))
                                                skip_lines(f, 14)
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
                                            skip_lines(f, 9)
                                            line = next(f)
                                            split_RSRP0 = line.split(RSRP_0)
                                            RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                            RSRP00 =float(RSRP00) - 30.0 
                                            RSRP0.append(float(RSRP00))
                                            line = next(f)
                                            if RSRP_1 in line:
                                                split_RSRP1 = line.split(RSRP_1)
                                                RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                RSRP11 =float(RSRP11) - 30.0
                                                RSRP1.append(float(RSRP11))
                                                line = next(f)
                                                if RSRP_2 in line:
                                                    split_RSRP2 = line.split(RSRP_2)
                                                    RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                    RSRP22 =float(RSRP22) - 30.0
                                                    RSRP2.append(float(RSRP22))
                                                    skip_lines(f, 2)
                                                    line = next(f)
                                                    if RSRP_3 in line:
                                                        split_RSRP3 = line.split(RSRP_3)
                                                        RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                        RSRP33 =float(RSRP33) - 30.0
                                                        RSRP3.append(float(RSRP33))
                                                        skip_lines(f, 14)
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
                                                    skip_lines(f, 9)
                                                    line = next(f)
                                                    split_RSRP0 = line.split(RSRP_0)
                                                    RSRP00 = split_RSRP0[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix
                                                    RSRP00 =float(RSRP00) - 30.0 
                                                    RSRP0.append(float(RSRP00))
                                                    line = next(f)
                                                    if RSRP_1 in line:
                                                        split_RSRP1 = line.split(RSRP_1)
                                                        RSRP11 = split_RSRP1[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                        RSRP11 =float(RSRP11) - 30.0
                                                        RSRP1.append(float(RSRP11))
                                                        line = next(f)
                                                        if RSRP_2 in line:
                                                            split_RSRP2 = line.split(RSRP_2)
                                                            RSRP22 = split_RSRP2[1].strip().replace(' dBm','')  # Remove the 'dBm' suffix 
                                                            RSRP22 =float(RSRP22) - 30.0
                                                            RSRP2.append(float(RSRP22))
                                                            skip_lines(f, 2)
                                                            line = next(f)
                                                            if RSRP_3 in line:
                                                                split_RSRP3 = line.split(RSRP_3)
                                                                RSRP33 = split_RSRP3[1].strip().replace(' dBm','')  # Remove the 'dB' suffix 
                                                                RSRP33 =float(RSRP33) - 30.0
                                                                RSRP3.append(float(RSRP33))
                                                                skip_lines(f, 14)
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
                delta_lon = lon2_rad - lon1_rad
                delta_lat = lat2_rad - lat1_rad

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

# print('Times:', times)
print('cellTimes:', cellTimes)
print('SINR Rx[0] :', SINR0)
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
print('The Final Latitude length:', len(latitude_list))
print('The Final Longitude length:', len(longitude_list))
print('The Final distance length:', len(distance_list))

def calculate_averages(lst):
    averages = []
    for i in range(0, len(lst), 5):
        chunk = lst[i:i+5]  # Extract a chunk of 5 elements
        avg = sum(chunk) / len(chunk)  # Calculate the average
        averages.append(avg)  # Add the average to the new list
    return averages

y00 = calculate_averages(SINR0)
y11 = calculate_averages(SINR1)
y22 = calculate_averages(SINR2)
y33 = calculate_averages(SINR3)
x00 = calculate_averages(distance_list)
print('y33:', len(y33))


# Perform polynomial regression with cross-validation
X = np.array(x00).reshape(-1, 1)

# Reshape the data to match the expected input shape for scikit-learn
y0 = np.array(y00)
y1 = np.array(y11)
y2 = np.array(y22)
y3 = np.array(y33)

# Scale the features
scaler = StandardScaler()

# Scale the input data
X_scaled = scaler.fit_transform(X)

# Define the gradient descent with regularization function
def gradient_descent_regularized(X, y, alpha, lambda_, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for i in range(iterations):
        h = X @ theta
        gradient = (1 / m) * X.T @ (h - y) + (lambda_ / m) * theta
        theta = theta - alpha * gradient
    return theta



# Generate the polynomial features
poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_poly = poly_features.fit_transform(X_scaled)

# Perform gradient descent with regularization
theta0 = gradient_descent_regularized(X_poly, y0[:, np.newaxis],alpha, lambda_, iterations)
theta1 = gradient_descent_regularized(X_poly, y1[:, np.newaxis],alpha, lambda_, iterations)
theta2 = gradient_descent_regularized(X_poly, y2[:, np.newaxis],alpha, lambda_, iterations)
theta3 = gradient_descent_regularized(X_poly, y3[:, np.newaxis],alpha, lambda_, iterations)

# Generate the predictions
X_new = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y0_pred = X_new_poly @ theta0
y1_pred = X_new_poly @ theta1
y2_pred = X_new_poly @ theta2
y3_pred = X_new_poly @ theta3

# Create a figure and axes objects
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot in the first subplot
axs[0, 0].scatter(X_scaled, y0, color='b', marker='x', label='Measured')
axs[0, 0].plot(X_new, y0_pred, color='r', label='Predicted')
axs[0, 0].set_xlabel('Scaled Distance')
axs[0, 0].set_ylabel('SINR Rx[0] in db')
axs[0, 0].set_title('SINR Rx[0] vs. Distance')
axs[0, 0].legend(loc='upper right')

# Plot in the second subplot
axs[0, 1].scatter(X_scaled, y1, color='b', marker='x', label='Measured')
axs[0, 1].plot(X_new, y1_pred, color='r', label='Predicted')
axs[0, 1].set_xlabel('Scaled Distance')
axs[0, 1].set_ylabel('SINR Rx[1] in db')
axs[0, 1].set_title('SINR Rx[1] vs. Distance')
axs[0, 1].legend(loc='upper right')

# Plot in the third subplot
axs[1, 0].scatter(X_scaled, y2, color='b', marker='x', label='Measured')
axs[1, 0].plot(X_new, y2_pred, color='r', label='Predicted')
axs[1, 0].set_xlabel('Scaled Distance')
axs[1, 0].set_ylabel('SINR Rx[2] in db')
axs[1, 0].set_title('SINR Rx[2] vs. Distance')
axs[1, 0].legend(loc='upper right')

# Plot in the fourth subplot
axs[1, 1].scatter(X_scaled, y3, color='b', marker='x', label='Measured')
axs[1, 1].plot(X_new, y3_pred, color='r', label='Predicted')
axs[1, 1].set_xlabel('Scaled Distance')
axs[1, 1].set_ylabel('SINR Rx[3] in db')
axs[1, 1].set_title('SINR Rx[3] vs. Distance')
axs[1, 1].legend(loc='upper right')

fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()
