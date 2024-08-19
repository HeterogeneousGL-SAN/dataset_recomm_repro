import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mes',
                    choices=['mes','pubmed_kcore','pubmed'])

parser.add_argument('--path_dataset', type=str)

args = parser.parse_args()
basepath = args.path_dataset

def write_results():

    files = os.listdir(basepath+'/')
    results = open(basepath+'_all_final_dec.txt','w')
    for file in files:
        if  file.endswith('tsv'):
            f = open(basepath + '/' + file, 'r')

            lines = f.readlines()


            total_T = 0
            total_G = 0
            total_N = 0
            total_prec = 0
            total_rec = 0
            title = '.'.join(file.split('.')[0:-1])
            for line in lines:
                values = line.split()
                T = float(values[1])
                G = float(values[2])
                N = float(values[3])
                prec = float(values[5])
                rec = float(values[4])
                total_prec += prec
                total_rec += rec

                total_T += T
                total_G += G
                total_N += N
            if total_N != 0:
                precision = total_T/total_N
            else:
                precision = 0
            if total_G != 0:
                recall = total_T/total_G
            else:
                recall = 0
            total_prec = total_prec/len(lines)
            total_rec = total_rec/len(lines)
            if precision != 0 and recall != 0:
                F1 = (precision*recall)/(precision+recall)*2
            else:
                F1 = 0
            l = title + '\t' + str(precision) + '\t' + str(recall) +'\t' + str(F1)+'\t'+str(total_prec) + '\t'+str(total_rec) +'\n'
            results.write(l)

if __name__ == '__main__':
    write_results()
