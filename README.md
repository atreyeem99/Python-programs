# obminimize
```
parser = argparse.ArgumentParser(description='Some versions of obabel (with --minimize) or obminimize show weird behaviour when working with multiple XYZs collected in a single file. Sometimes the outputt is pdb with fewer significant figures. This python code offers a solution. It takes as input an XYZ file containing atomic coordinates of multiple molecules and optimizes each tightly using obabel. Thelt options are set as \'--ff UFF --sd --c 1e-6 --n 10000\'. Feel free to experiment.')

parser.add_argument('Nmol', type=int, help='Number of molecules')
parser.add_argument('XYZinit', type=str, help='Name of the XYZ file with initial coordinates')
parser.add_argument('XYZfina', type=str, help='Name of the XYZ file with final coordinates')

args = parser.parse_args()

Nmol=args.Nmol
XYZinit=args.XYZinit
XYZfina=args.XYZfina

filedir = os.getcwd()

os.system(f'rm {XYZfina}')

geom_file = open(XYZinit, 'r')

for imol in range(Nmol):

    line = geom_file.readline().strip()

    if line:

        Nat = int(line)
        title = geom_file.readline().strip()
        print(Nat, title)

        inputfile= open('obabel_cho.xyz', 'w')

        inputfile.write(f'{Nat}\n')
        inputfile.write(f'{title}\n')

        for iat in range(1, Nat + 1):
            line = geom_file.readline().split()
            sym=line[0]
            R=[float(line[1]), float(line[2]), float(line[3])]
            inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

        inputfile.close()- 

        os.system(f'obabel cho_obabel.xyz -oxyz -O cho_UFF_tight.xyz --minimize --ff UFF --sd --c 1e-6 --n 10000')

        os.system(f'cat cho_UFF_tight.xyz >> {XYZfina}')
        os.system(f'rm cho_obabel.xyz cho_UFF_tight.xyz')

geom_file.close()
```
# prepinp_geom
```
import os

Nmol = 4
geomfile='comp_UFF_tight.xyz'

filedir = os.getcwd()

geom_file = open(geomfile, 'r')

for imol in range(Nmol):

    line = geom_file.readline().strip()

    if line:

        Nat = int(line)
        title = geom_file.readline().strip()
        print(Nat, title)

        inputfile= open('geom_UFF.xyz', 'w')

        inputfile.write(f'{Nat}\n')
        inputfile.write(f'{title}\n')

        for iat in range(1, Nat + 1):
            line = geom_file.readline().split()
            sym=line[0]
            R=[float(line[1]), float(line[2]), float(line[3])]
            inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

        inputfile.close()

        os.mkdir(os.path.join(filedir, title))

        os.system(f'cp Geoopt_wB97XD3_def2TZVP.com geom_UFF.xyz {title}/')

geom_file.close()

```
# prepinp_dft
```
import os

Nmol = 5

geomfile='comp_UFF_tight.xyz' 
geom_file = open(geomfile, 'r')

geomfiledft='comp_DFT_S0.xyz'
geom_file_dft = open(geomfiledft, 'r')

filedir = os.getcwd()

for imol in range(Nmol):

    line = geom_file.readline().strip()  # Nat

    if line:

        # read 2 lines from geomfiledft
        title = geom_file_dft.readline().strip() # 1st line in DFT xyz, Nat
        title = geom_file_dft.readline().strip() # 2nd line in DFT xyz, title

        Nat = int(line)
        title = geom_file.readline().strip() # title from UFF file
        print(Nat, title)

        inputfile= open('geom_DFT_S0.xyz', 'w')

        inputfile.write(f'{Nat}\n')
        inputfile.write(f'{title}\n')

        for iat in range(1, Nat + 1):
            line = geom_file.readline().split()   # read coordinates from UFF, never used
            line = geom_file_dft.readline().split() # read coordinates from DFT file
            sym=line[0]
            R=[float(line[1]), float(line[2]), float(line[3])]
            inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

        inputfile.close()

        #os.mkdir(os.path.join(filedir, title))

        os.system(f'cp TDDFT_wB97XD3_def2TZVP.com geom_DFT_S0.xyz {title}/')

geom_file.close()
```
# Generate a plot of txt file by python
```
python3
import matplotlib.pyplot as plt
import numpy as np
A= np.load.txt("textfilename.txt")
plt.plot(A)
plt.grid()
plt.show()
```
# use different functional groups on one compound
```
string_template = "C(=O)C({})=C({}){}"

# Words to fill in the brackets
groups = ['', 'C', 'N','O','F','CC','C=C','C#C','C#N','C=N','CN','CO','C(=O)','CF','OC']

Ngrps=len(groups)

file1=open('acyclic_aldehyde.smi','w')

for i in range(Ngrps):
    for j in range(Ngrps):
        for k in range(Ngrps):
            groupi=groups[i]
            groupj=groups[j]
            groupk=groups[k]
            if len(groupi) != 0:
                groupi='('+groupi+')'
            if len(groupj) != 0:
                groupj='('+groupj+')'
            mol='C(=O)C'+groupi+'=C'+groupj+groupk
            name='acyclic_ald_'+str(i)+'_'+str(j)+'_'+str(k)
            file1.write(mol+' '+name+'\n')
    

file1.close()
```
