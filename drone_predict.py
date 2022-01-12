import pandas as pd
import sys
import os
import joblib  # used to save / import models
import argparse


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


clasfs = ['G1', 'G2', 'G3', 'D1', 'D2', 'D3']
align = ['HH', 'VV']
drones = ['F450', 'HELICOPTER', 'MAVIC', 'PARROT',
          'P4P', 'HEXA', 'M100', 'WALKERA', 'Y600']
freq = [26, 28, 30, 32, 34, 36, 38, 40]

parser = argparse.ArgumentParser(
    description='Drone Recognition from RCS signatures     -   suggested classifier per drones: KNN, 32GHz ')

group1 = parser.add_argument_group(
    'Data from dataset', 'Given freq, name and alignment, takes data from the dataset automatically')
group2 = parser.add_argument_group(
    'Manual Data', 'Manually give freq, mean, max')


parser.add_argument("-c", "--classifiers", choices=clasfs,
                    help="choose classifier: <G1> Decision Tree, <G2> KNN, <G3> SGDC,  <D1> Decision Tree, <D2> KNN, <D3> SGDC", metavar='', required=True)
group1.add_argument("-a", "--alignment", choices=align,
                    help="choose alignment: HH or VV", metavar='')
group1.add_argument("-m", "--model", choices=drones,
                    help="choose drone - F450, HELICOPTER, MAVIC, PARROT, P4P, HEXA, M100, WALKERA, Y600", metavar='')
group1.add_argument("-f", "--frequency", choices=freq, type=int,
                    help="choose frequency: 26, 28, 30, 32, 34, 36, 38, 40 [GHz]", metavar=' ')
group2.add_argument("-od", "--override", nargs='+', type=float,
                    help="manual override. Write frequency, mean, max - it works without any check")

args = parser.parse_args()


# ARGPARSE classifiers
if (args.classifiers != None):
    if (args.classifiers == 'G1') or (args.classifiers == 'G2') or (args.classifiers == 'G3'):
        file_path = "/models/group recognition/"
    else:
        file_path = "/models/drone recognition/"

    if args.classifiers == 'G1':
        model = '6-Decision Tree Classifier frequency - group'
    if args.classifiers == 'G2':
        model = '6-K Nearest Neighbors frequency - group'
    if args.classifiers == 'G3':
        model = '6-SGDC Classifier frequency - group'
    if args.classifiers == 'D1':
        model = '7-Decision Tree Classifier frequency - drone'
    if args.classifiers == 'D2':
        model = '7-K Nearest Neighbors frequency - drone'
    if args.classifiers == 'D3':
        model = '7-SGDC Classifier frequency - drone'


# Getting base directory
base_dir = os.path.dirname(os.path.realpath(__file__))
filepath = base_dir + file_path

# Import dataframe
df = pd.read_excel('table2.ods')

# TODO TEST ARGS NOT NULL != override.
loaded_model = joblib.load(str(filepath + model + ".sav"))

if(args.override is not None) and ((args.alignment is not None) or (args.model is not None) or (args.frequency is not None)):
    print(bcolors.FAIL +
          "ERROR: use -a, -m, -f together, -od alone. -c is required." + bcolors.ENDC)
    sys.exit(1)
elif (args.override != None) and (len(args.override) != 3):
    print(bcolors.FAIL + "3 values are needed. Exiting" + bcolors.ENDC)
    sys.exit(1)
else:
    print(bcolors.OKCYAN + "OK" + bcolors.ENDC)

# args -a, -m, -f for not none
if (args.override == None):
    test = df.loc[(df['Pos'] == args.alignment) & (
        df['Model'] == args.model) & (df['f[GHz]'] == args.frequency)]
    X_test = test.drop(columns=['Model', 'Pos', 'std', 'Group'])

    # TEST HARDCODED
    # ['P4P']
    # X_test = [[26,-15.7,0.1]]
    print("predicted:  " + bcolors.BOLD + bcolors.OKGREEN +
          str(loaded_model.predict(X_test)[0]) + bcolors.ENDC)
else:
    X_test = [[args.override[0], args.override[1], args.override[2]]]
    print("predicted:  " + bcolors.BOLD + bcolors.OKGREEN +
          str(loaded_model.predict(X_test)[0]) + bcolors.ENDC)
