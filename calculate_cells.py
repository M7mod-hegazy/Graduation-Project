# import matplotlib.pyplot as plt
import re
import math


# Define a function to skip lines
def skip_lines(file, num_lines):
    for _ in range(num_lines):
        next(file)

# Define the pattern to match for the start of the block
pattern1 = '*** Mode Report Message type: ML1 Serving Cell Measurement Result'
pattern2 = 'Device: EQ1'
pattern3 = 'CurrentRAT: LTE'
patt0='Residual Frequency Error :'
cell_Patt = '  [0 ] :'
cell_Patt1 = '  [1 ] :'
cell_Patt2 = '  [2 ] :'
cell = 'Physical Cell ID : '

# Define the strings to search for within the block
search_time = 'Time : '



# Initialize empty lists to store the values of interest
times = []
cell_list = []

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
                            if int(TimeParts[1]) >= 12 and int(TimeParts[2]) >= 20 and int(smallTime[0]) >= 4 and int(smallTime[1]) >= 247 :
                                time = split_line[1].strip()
                                times.append(time)
                                skip_lines(f, 22)
                                line =next(f)

                                if cell_Patt in line:
                                    line = next(f)
                                    split_line0 = line.split(cell)
                                    cell_list.append(int(split_line0[1]))
                                    skip_lines(f, 39)
                                    line = next(f)
                                    if cell_Patt1 in line:
                                        line = next(f)
                                        split_line0 = line.split(cell)
                                        cell_list.append(int(split_line0[1]))
                                        skip_lines(f, 39)
                                        line = next(f)
                                        if cell_Patt2 in line:
                                            line = next(f)
                                            split_line0 = line.split(cell)
                                            cell_list.append(int(split_line0[1]))
                                            line = next(f)



def count_occurrences(lst):
    occurrence_count = {}
    for element in lst:
        if element in occurrence_count:
            occurrence_count[element] += 1
        else:
            occurrence_count[element] = 1
    return occurrence_count

result  = count_occurrences(cell_list)
# print(f'cells : {cell_list}')
print(result)
# print(f'non_repeat_cells : {unique_cells}')