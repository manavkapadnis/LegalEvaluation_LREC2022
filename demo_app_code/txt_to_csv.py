# -*- coding: utf-8 -*-
"""txt to csv.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18lhzaXxvtU4cTR6_TiEHMm5Mx0Y9hriY
"""

import csv
import os

header = ["Text"]

with open('data1.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(header)

    for x in range(1, 65):
        filepath = os.path.normpath(r'.\summary_{}.txt'.format(x))

        with open(filepath, 'r', newline='') as f_text:
            csv_text = csv.reader(f_text, delimiter=':', skipinitialspace=True)
            csv_output.writerow(csv_text)