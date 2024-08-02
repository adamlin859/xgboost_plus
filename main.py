import sys
from synthetic_experiment import expriment1, expriment2
from pbc_experiment import experiment3 

if __name__ == "__main__":
    n = len(sys.argv)

    if n != 2:
        print(f'ERROR: Please run the command using the following format: python3 main.py experimentName')
        sys.exit(1)

    if sys.argv[1] == 'experiment1':
        expriment1()
    elif sys.argv[1] == 'experiment2':
        expriment2()
    elif sys.argv[1] == 'experiment3':
        experiment3()