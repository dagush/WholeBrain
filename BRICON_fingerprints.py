# =============================================================
#  Simple text to check we can read all fingerprints
# =============================================================
import os, re

basePath = 'Data_Raw/julichbrain_data/receptor_fingerprints'

dirs = [x[0] for x in os.walk(basePath)]
print('Areas: ' + ' '.join([os.path.split(a)[1].split("_")[0] for a in dirs[1:]]))
for dir in dirs:
    dir = os.path.normpath(dir)
    print('\n#########################################################')
    print(f'checking: {dir} -> {os.path.split(dir)[1].split("_")[0]}')
    for file in os.listdir(dir):
        if file.endswith(".tsv"):
            num_lines = sum(1 for line in open(os.path.join(dir,file), "r"))
            if num_lines > 6:  # 5 receptors + 1 title -> these are repeated in other files in the same directory
                print(f'   found: {file}')
                f = open(os.path.join(dir,file), "r")
                f.readline()
                for x in f:
                    x = x.strip("\n")
                    print(x.split("\t"), end=' ')
                print()
            else:
                print(f'   found: {file} -> DISCARDED! (receptors = {num_lines-1})')



