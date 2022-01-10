import pandas as pd
import sys, os
import joblib # used to save / import models
import argparse

clasfs = ['G1', 'G2', 'G3', 'D1', 'D2', 'D3']
align = ['HH', 'VV']
drones = ['F450','HELICOPTER','MAVIC','PARROT','P4P','HEXA','M100','WALKERA','Y600']
freq = [26,28,30,32,34,36,38,40]

parser = argparse.ArgumentParser(description='Drone Recognition from RCS signatures     -   suggested classifier per drones: KNN, 32GHz ')
parser.add_argument("-c", "--classifiers", choices = clasfs, help = "choose classifier: <G1> Decision Tree, <G2> KNN, <G3> SGDC,  <D1> Decision Tree, <D2> KNN, <D3> SGDC",metavar='')
parser.add_argument("-a", "--alignment", choices = align, default = align[1], help = "choose alignment: HH or VV",metavar = '')
parser.add_argument("-m", "--model", choices = drones, default = drones[0], help = "choose drone - F450, HELICOPTER, MAVIC, PARROT, P4P, HEXA, M100, WALKERA, Y600", metavar='')
parser.add_argument("-f", "--frequency", choices = freq, type=int, default = freq[3], help = "choose frequency: 26, 28, 30, 32, 34, 36, 38, 40 [GHz]",metavar =' ')
parser.add_argument("-od", "--override", nargs='+', type=float, help = "manual override. Write frequency, mean, max - it works without any check")

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

if args.override == None:
    test = df.loc[(df['Pos'] == args.alignment) & (df['Model'] == args.model) & (df['f[GHz]'] == args.frequency)]
    X_test = test.drop(columns=['Model', 'Pos','std', 'Group'])

    # TEST HARDCODED
    # ['P4P']
    # X_test = [[26,-15.7,0.1]]
    print(loaded_model.predict(X_test))
    
else:
    if len(args.override) != 3:
        print("3 values are needed. Exiting")
        sys.exit(1)
    else:
        X_test = [[args.override[0], args.override[1], args.override[2]]]
        print(loaded_model.predict(X_test))

    
