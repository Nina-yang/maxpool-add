import argparse
import os
import glob
import numpy as np
import re

def get_single_exp_status(fname):
    record = []
    with open(fname) as f:
        data = f.read().splitlines()
    for i in range(len(data)):
        if not i & 0x1:
            continue
        # cnt = re.findall('-?\d+.?\d+', data[i])
        cnt = float(data[i].split(' ')[3])
        record.append(cnt)
    return record
        

def generate_report(in_directory, out_directory):
    d = dict()
    for path in glob.glob(os.path.join(in_directory,"*.txt")):
        fname = os.path.split(path)[-1][:-4]
        d[fname] = get_single_exp_status(path)
    
    for key, value in d.items():
        # print(value)
        print(key, "%.3f" % np.mean(value), "%.3f" % np.std(value))
        
                           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--in_directory', type = str, default='./report', help='Directory to get input files')
    parser.add_argument('-o','--out_directory', type = str, default='./report', help='Directory to save output reports')
    args = parser.parse_args()
 
    generate_report(args.in_directory, args.out_directory)