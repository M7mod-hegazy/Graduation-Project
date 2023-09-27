n line:
                                                        split_RSRQ1 = line.split(RSRQ_1)
                                                        RSRQ11 = split_RSRQ1[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                        RSRQ11 =float(RSRQ11) 
                                                        RSRQ1_TM8.append(float(RSRQ11))
                                                        line = next(f)
                                                        if RSRQ_2 in line:
                                                            split_RSRQ2 = line.split(RSRQ_2)
                                                            RSRQ22 = split_RSRQ2[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                            RSRQ22 =float(RSRQ22) 
                                                            RSRQ2_TM8.append(float(RSRQ22))
                                                            
                                                            line = next(f)
                                                            if RSRQ_3 in line:
                                                                split_RSRQ3 = line.split(RSRQ_3)
                                                                RSRQ33 = split_RSRQ3[1].strip().replace(' dB','')  # Remove the 'dB' suffix 
                                                                RSRQ33 =float(RSRQ33) 
                                                                RSRQ3_TM8.append(float(RSRQ33))
                                                                skip_lines(f, 8)
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