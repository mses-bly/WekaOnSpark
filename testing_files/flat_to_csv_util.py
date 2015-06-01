import csv
import itertools

inputFile = raw_input('Input file name: ')
outputFile = raw_input('Output file name: ')
with open(inputFile, 'r') as in_file:
    splitted = (line.split() for line in in_file)
    with open(outputFile, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(splitted)