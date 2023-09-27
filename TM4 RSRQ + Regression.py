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
poly_degree = 3
alpha=0.01
lambda_=0.01
iterations=10000

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
                                        line = next(f)
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
                                                line = next(f)
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

          
# Print the lists values

print('Times:', times[-10:])
print('cellTimes:', cellTimes[-10:])
print('SINR Rx[0] :', SINR0[-10:])
print('RSRQ Rx[0] :', RSRQ0[-10:])
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

# Define the gradient descent with regularization function
def gradient_descent_regularized(X, y, alpha, lambda_, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for i in range(iterations):
        h = X @ theta
        gradient = (1 / m) * X.T @ (h - y) + (lambda_ / m) * theta
        theta = theta - alpha * gradient
    return theta

# Generate the polynomial features for each x
poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
x0_poly = poly_features.fit_transform(x0_scaled)
x1_poly = poly_features.fit_transform(x1_scaled)
x2_poly = poly_features.fit_transform(x2_scaled)
x3_poly = poly_features.fit_transform(x3_scaled)

# Perform gradient descent with regularization for each y
theta0 = gradient_descent_regularized(x0_poly, y0[:, np.newaxis], alpha, lambda_, iterations)
theta1 = gradient_descent_regularized(x1_poly, y1[:, np.newaxis], alpha, lambda_, iterations)
theta2 = gradient_descent_regularized(x2_poly, y2[:, np.newaxis], alpha, lambda_, iterations)
theta3 = gradient_descent_regularized(x3_poly, y3[:, np.newaxis], alpha, lambda_, iterations)

# Generate the predictions
x_new0 = np.linspace(x0_scaled.min(), x0_scaled.max(), 100).reshape(-1, 1)
x_new_poly0 = poly_features.transform(x_new0)

x_new1 = np.linspace(x1_scaled.min(), x1_scaled.max(), 100).reshape(-1, 1)
x_new_poly1 = poly_features.transform(x_new1)

x_new2 = np.linspace(x2_scaled.min(), x2_scaled.max(), 100).reshape(-1, 1)
x_new_poly2 = poly_features.transform(x_new2)

x_new3 = np.linspace(x3_scaled.min(), x3_scaled.max(), 100).reshape(-1, 1)
x_new_poly3 = poly_features.transform(x_new3)

y0_pred = x_new_poly0 @ theta0
y1_pred = x_new_poly1 @ theta1
y2_pred = x_new_poly2 @ theta2
y3_pred = x_new_poly3 @ theta3


# Create a figure and axes objects
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot in the first subplot
axs[0, 0].scatter(x0_scaled, y0, color='b', marker='x', label='Measured')
axs[0, 0].plot(x_new0, y0_pred, color='r', label='Predicted')
# axs[0, 0].set_xlim(-3.5, 2.5)
axs[0, 0].set_xlabel('Scaled RSRQ Rx[0]')
axs[0, 0].set_ylabel('SINR Rx[0] in db')
axs[0, 0].set_title('SINR Rx[0] vs. RSRQ Rx[0]')
axs[0, 0].legend(loc='lower right')

# Plot in the second subplot
axs[0, 1].scatter(x1_scaled, y1, color='b', marker='x', label='Measured')
axs[0, 1].plot(x_new1, y1_pred, color='r', label='Predicted')
# axs[0, 1].set_xlim(-3.5, 2.5)
axs[0, 1].set_xlabel('Scaled RSRQ Rx[1]')
axs[0, 1].set_ylabel('SINR Rx[1] in db')
axs[0, 1].set_title('SINR Rx[1] vs. RSRQ Rx[1]')
axs[0, 1].legend(loc='lower right')

# Plot in the third subplot
axs[1, 0].scatter(x2_scaled, y2, color='b', marker='x', label='Measured')
axs[1, 0].plot(x_new2, y2_pred, color='r', label='Predicted')
# axs[1, 0].set_xlim(-3.5, 2.5)
axs[1, 0].set_xlabel('Scaled RSRQ Rx[2]')
axs[1, 0].set_ylabel('SINR Rx[2] in db')
axs[1, 0].set_title('SINR Rx[2] vs. RSRQ Rx[2]')
axs[1, 0].legend(loc='lower right')

# Plot in the fourth subplot
axs[1, 1].scatter(x3_scaled, y3, color='b', marker='x', label='Measured')
axs[1, 1].plot(x_new3, y3_pred, color='r', label='Predicted')
# axs[1, 1].set_xlim(-3.5, 2.5)
axs[1, 1].set_xlabel('Scaled RSRQ Rx[3]')
axs[1, 1].set_ylabel('SINR Rx[3] in db')
axs[1, 1].set_title('SINR Rx[3] vs. RSRQ Rx[3]')
axs[1, 1].legend(loc='lower right')

fig.suptitle(pltAddress, fontsize=14, fontweight='bold')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()
