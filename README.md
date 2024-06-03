# obminimize : for creating folders to merge the xyz of different structures to one UFF xyz file
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
 - to run the program give the command
   ```
   python3 obminimize.py nmol geom.xyz geom_UFF.xyz
   ```
   where nmol is number of molecules
# obminimize (sdf to xyz)
```
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Some versions of obabel (with --minimize) or obminimize show weird behaviour when working with multiple XYZs collected in a single file. Sometimes the output format is pdb with fewer significant figures. This python code offers a solution. It takes as input an SDF file containing atomic coordinates of multiple molecules, converts it to XYZ, and optimizes each molecule using obabel. The default options are set as \'--ff UFF --sd --c 1e-6 --n 10000\'. Feel free to experiment.')

parser.add_argument('Nmol', type=int, help='Number of molecules')
parser.add_argument('SDFinit', type=str, help='Name of the SDF file with initial coordinates')
parser.add_argument('XYZfina', type=str, help='Name of the XYZ file with final coordinates')

args = parser.parse_args()

Nmol=args.Nmol
SDFinit=args.SDFinit
XYZfina=args.XYZfina

filedir = os.getcwd()

os.system(f'rm {XYZfina}')

# Convert SDF to XYZ
os.system(f'obabel {SDFinit} -oxyz -O {SDFinit[:-4]}.xyz')

geom_file = open(f'{SDFinit[:-4]}.xyz', 'r')

for imol in range(Nmol):

    line = geom_file.readline().strip()

    if line:

        Nat = int(line)
        title = geom_file.readline().strip()
        print(Nat, title)

        inputfile= open('geom.xyz', 'w')

        inputfile.write(f'{Nat}\n')
        inputfile.write(f'{title}\n')

        for iat in range(1, Nat + 1):
            line = geom_file.readline().split()
            sym=line[0]
            R=[float(line[1]), float(line[2]), float(line[3])]
            inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

        inputfile.close()

        os.system(f'obabel geom.xyz -oxyz -O geom_tmp.xyz --minimize --ff UFF --sd --c 1e-6 --n 10000')

        os.system(f'cat geom_tmp.xyz >> {XYZfina}')
        os.system(f'rm geom.xyz geom_tmp.xyz')

geom_file.close()
```
- ` python3 obminimize nmol file.sdf file.xyz`
`python3 obminimize nmol file.sdf file.xyz`
# prepinp_geom : to create folders with opt input file and xyz coordinates
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

# prepinp_dft: to create folders with tddft input file and xyz coordinates
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
# use different functional groups on one compound using smiles. All probable combinations are obtained
```
string_template = "C(=O)C({})=C({}){}"


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

```
#string_template = "C1{}C{}=C{}NC(=O)N=1"

groups = ['','N','C','F'] 

Ngrps=len(groups)

file1=open('1H_2pyramidinone.smi','w')

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
            if len(groupk) != 0:
                groupk='('+groupk+')' 
            mol='C1'+groupi+'C'+groupj+'=C'+groupk+'NC(=O)N=1'
            name='1H_2pyramidinone_'+str(i)+'_'+str(j)+'_'+str(k)
            file1.write(mol+' '+name+'\n')
    

file1.close()
```
```
g_r_rt=-192.93949117
dg_r_rt=0.05818173
dg_r_1000K=-0.03679196
g_r_1000K= g_r_rt-dg_r_rt+dg_r_1000K

se_r_dlpno=-192.776810360853
g_r_1000_dlpno=se_r_dlpno+dg_r_1000K

g_ts_rt=-192.83678263
dg_ts_rt=0.05145363
dg_ts_1000K=-0.05211098
g_ts_1000K= g_ts_rt-dg_ts_rt+dg_ts_1000K

se_ts_dlpno=-192.6635437519
g_ts_1000_dlpno=se_ts_dlpno+dg_ts_1000K

au2kcm=au2kcm=627.5096080305927

g_barr_1000_dft=(g_ts_1000K-g_r_1000K)*au2kcm
g_barr_1000_dlpno=(g_ts_1000_dlpno-g_r_1000_dlpno)*au2kcm

print(g_barr_1000_dft)
print(g_barr_1000_dlpno)
```

# to plot the data from a txt file 
```
import matplotlib.pyplot as plt

def plot_column(input_file, column_index):
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
            y_values = [float(line.split()[column_index]) for line in lines]
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Plot the data
    plt.plot(y_values, label=f'Column {column_index}')
    plt.xlim(1000,2000)
    plt.title(f'Plot of Column {column_index} from Input File')
    plt.xlabel('Data Point Index')
    plt.ylabel(f'Column {column_index}')
    plt.legend()
    plt.show()

input_file_path = '/home/atreyee/project/orca_manual/esd/benzene_esd.txt'  # Replace with your actual file path
column_to_plot = 1  # Replace with the desired column index
plot_column(input_file_path, column_to_plot)

```
# To calculate mean standard error, mean absolute error and standard deviation error
```
import csv
import numpy as np

def read_column(filename, column_index):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                value = float(row[column_index])
                data.append(value)
            except (ValueError, IndexError):
                pass
    return data

def calculate_mean_std_error(data1, data2):
    mse = np.mean(data1-data2)
    mae = np.mean(np.abs(data1-data2))
    sde = np.std(data1-data2)
    
    return mse,mae,sde

def main():
    # Replace 'file1.csv' and 'file2.csv' with the actual file names
    file1 = 'file1.csv'
    file2='file2.csv'
    
    # Assuming the 3rd column is at index 2 (0-based index)
    column_index = 2

    # Read the 3rd column from each file
    data1 = read_column(file1, column_index)
    data2 = read_column(file2, column_index)

    # Calculate mean and standard error
    data1=np.array(data1)
    data2=np.array(data2)
    mse,mae,sde = calculate_mean_std_error(data2, data1)
    print(data1)
    print(data2)
   
   

    print(mse,mae,sde)

    return

if __name__ == "__main__":
    main()

```
# To read the csv filenames from a file and to find the errors mse mae and sde
```
import csv
import numpy as np

def read_column(filename, column_index):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                value = float(row[column_index])
                data.append(value)
            except (ValueError, IndexError):
                pass
    return data

def calculate_mean_std_error(data1, data2):
    mse = np.mean(data1 - data2)
    mae = np.mean(np.abs(data1 - data2))
    sde = np.std(data1 - data2)

    return mse, mae, sde

def main():
    list_file = 'list.txt'
    column_index = 2
    file1 = 'a.csv'
    data1 = read_column(file1, column_index)
    with open(list_file, 'r') as file:
        # Read the list of CSV file names
        files=[]
        m=[]
        for line in file:
            m.append(line.strip())
            files.append('stringa'+line.strip()+'stringb.csv')
        #files = [line.strip() for line in file]
   
     
    for i in range(len(files)):
            # Read the 3rd column from each file
            data2 = read_column(files[i], column_index)

            # Calculate mean and standard error
            data1 = np.array(data1)
            data2 = np.array(data2)
            mse, mae, sde = calculate_mean_std_error(data2, data1)
            output = "{xc:20s}{val1:20.3f} {val2:20.3f} {val3:20.3f}"
            print(output.format(xc=m[i],val1=mse, val2=mae, val3=sde))


if __name__ == "__main__":
     main()
```
# python program to read a xyz file and calculate the distance between the coordinates of the 6th atom and all the atoms
```
import math

def read_xyz_file(file_path):
    atoms = []
    with open(file_path, 'r') as file:
        num_atoms = int(file.readline())
        file.readline()  # Skip the comment line

        for _ in range(num_atoms):
            line = file.readline().split()
            atom_symbol, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
            atoms.append((atom_symbol, (x, y, z)))

    return atoms

def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def main():
    file_path = 'your_xyz_file.xyz'  # Replace with the path to your XYZ file
    atoms = read_xyz_file(file_path)

    if len(atoms) >= 6:
        sixth_atom = atoms[5]  # 0-based index, so the 6th atom is at index 5
        print(f"Coordinates of the 6th atom: {sixth_atom[1]}")

        for i, atom in enumerate(atoms):
            if i != 5:  # Skip the 6th atom
                distance = calculate_distance(sixth_atom, atom)
                print(f"Distance between the 6th atom and atom {i + 1}: {distance:.3f}")

    else:
        print("Not enough atoms in the file.")

if __name__ == "__main__":
    main()
```
# write a program to read a csv file which has the first column as the names of xyz files. Then go to ech xyz file and see which molecule has the wanted atom. Then calculate the distance between the coordinates of that atom and all the atoms, then sort the distances in ascending order. After that with the indexes of the 3 atoms with shortest distance, calculate the vector distance between the fluorine atom and the 3 other atoms. Then do the above program with angle calculation. Then plot the fifth column of csv file vs the calculated deviations.
```
import csv
import math
import matplotlib.pyplot as plt

def read_csv_file(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

def read_xyz_file(file_path):
    atoms = []
    with open(file_path+'.xyz', 'r') as file:
        num_atoms = int(file.readline())
        file.readline()  # Skip the comment line

        for _ in range(num_atoms):
            line = file.readline().split()
            atom_symbol, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
            atoms.append((atom_symbol, (x, y, z)))

    return atoms

def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def calculate_vector_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance_vector = (x2 - x1, y2 - y1, z2 - z1)
    return distance_vector

def calculate_angle(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a**2 for a in vector1))
    magnitude2 = math.sqrt(sum(b**2 for b in vector2))
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(max(-1, min(1, cos_theta)))  # Ensure the value is within [-1, 1] for acos
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def main():
    csv_file_path = 'top100.csv'  # Replace with the path to your CSV file
    data = read_csv_file(csv_file_path)
    print(data)
    
    deviations = []  # List to store deviations
    y_values = []  # List to store values for the y-axis

    for row in data:
        xyz_file = row[0]
        atoms = read_xyz_file(xyz_file)

        nitrogen_atoms = [atom for atom in atoms if atom[0] == 'N']

        if len(nitrogen_atoms) == 1:
            nitrogen_atom = nitrogen_atoms[0]
            print(f"\nXYZ File: {xyz_file}")
            print(f"Coordinates of the nitrogen atom: {nitrogen_atom[1]}")

            distances = []
            for i, atom in enumerate(atoms):
                if atom != nitrogen_atom:
                    distance = calculate_distance(nitrogen_atom, atom)
                    distances.append((i, distance))

            # Sort distances in ascending order
            sorted_distances = sorted(distances, key=lambda x: x[1])

            # Get the indexes of the 3 atoms with shortest distance
            closest_atoms_indexes = [index for index, _ in sorted_distances[:3]]

            # Calculate vector distances between the fluorine atom and the 3 closest atoms
            vectors = [calculate_vector_distance(nitrogen_atom, atoms[index]) for index in closest_atoms_indexes]

            angles = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    angle = calculate_angle(vectors[i], vectors[j])
                    angles.append(angle)

            average_angle = sum(angles) / len(angles)
            deviation_from_ideal = 109.5 - average_angle

            print(f"\nAverage angle: {average_angle:.2f} degrees")
            print(f"Deviation from (109.5 degrees): {deviation_from_ideal:.2f} degrees")

            deviations.append(deviation_from_ideal)
            y_values.append(float(row[4]))  # Assuming the 4th column is numeric for the y-axis

        #else:
         #   print(f"\nXYZ File: {xyz_file}")
          #  print("Error: More than one or no nitrogen atom found.")

    # Plotting
    plt.scatter(deviations, y_values, marker='o', color='blue')
    plt.title('Deviation vs. Y Values')
    plt.xlabel('Deviation from 109.5')
    plt.ylabel('Y Values')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
```
# With Plane to point distance
```
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def read_csv_file(csv_file_path):
    file_list = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            file_list.append(row)
    return file_list

def read_xyz_file(file_path):
   # print (file_path)
    atoms = []
    with open(file_path+'.xyz', 'r') as file:
        num_atoms = int(file.readline())
        file.readline()  # Skip the comment line

        for _ in range(num_atoms):
            line = file.readline().split()
            atom_symbol, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
            atoms.append((atom_symbol, (x, y, z)))

    return atoms

def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def calculate_vector_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance_vector = (x2 - x1, y2 - y1, z2 - z1)
    return distance_vector

def calculate_plane_normal(vector1, vector2):
    normal_vector = np.cross(vector1, vector2)
    return normal_vector

def calculate_distance_to_plane(point, plane_point, plane_normal):
    distance_to_plane = np.abs(np.dot(plane_normal, np.array(point) - np.array(plane_point)))
    return distance_to_plane

def main():
     csv_file_path = 'topall.csv'  # Replace with the path to your CSV file
     output_csv_path = 'distances_to_plane_output.csv'  # Replace with the desired output path
     data= read_csv_file(csv_file_path)
     #print(data)

     distances_to_plane = []  # List to store distances to the plane
     abc_values=[]
    #for xyz_file in xyz_files:
     for row in data:
        xyz_file = row[0]
        atoms = read_xyz_file(xyz_file)
        
        nitrogen_atoms = [atom for atom in atoms if atom[0] == 'N']
        
        if len(nitrogen_atoms) == 1:
            nitrogen_atom = nitrogen_atoms[0]

            distances = []
            for i, atom in enumerate(atoms):
                if atom != nitrogen_atom:
                    distance = calculate_distance(nitrogen_atom, atom)
                    distances.append((i, distance))

            # Sort distances in ascending order
            sorted_distances = sorted(distances, key=lambda x: x[1])

            # Get the indexes of the 3 atoms with the shortest distance
            closest_atoms_indexes = [index for index, _ in sorted_distances[:3]]

            # Calculate vector distances between the fluorine atom and the 3 closest atoms
            #vectors = [calculate_vector_distance(nitrogen_atom, atoms[index]) for index in closest_atoms_indexes]
            i1=closest_atoms_indexes[0]
            i2=closest_atoms_indexes[1]
            i3=closest_atoms_indexes[2]
            vector1=np.array(atoms[i2][1])-np.array(atoms[i1][1])
            vector2=np.array(atoms[i3][1])-np.array(atoms[i1][1])

            # Calculate the normal vector to the plane
            plane_normal = calculate_plane_normal(vector1, vector2)
            a=plane_normal[0]
            b=plane_normal[1]
            c=plane_normal[2]
            d=-np.dot(plane_normal,np.array(atoms[i1][1]))
            #print(d)
            x0=nitrogen_atom[1][0]
            y0=nitrogen_atom[1][1]
            z0=nitrogen_atom[1][2]
            distance_to_plane=np.abs(a*x0+b*y0+c*z0+d)/np.sqrt(a**2+b**2+c**2)

            # Calculate the distance from the fluorine atom to the plane
           # distance_to_plane = calculate_distance_to_plane(nitrogen_atom[1], atoms[closest_atoms_indexes[0]][1], plane_normal)
            #print(distance_to_plane)
            d1=sorted_distances[0][1]
            #print(d1)
            d2=sorted_distances[1][1]
            d3=sorted_distances[2][1]
            #print(d1,d2,d3)
            if d1<1.6 and d2<1.6 and d3<1.6 and float(row[4])<0.3 and distance_to_plane<0.05:
                print(d1,d2,d3,distance_to_plane,xyz_file,float(row[4]))
                distances_to_plane.append(distance_to_plane)
                #print(float(row))
                abc_values.append(float(row[4]))

    # Save distances to the plane to a CSV file
   # with open(output_csv_path, 'w', newline='') as output_csv:
    #    csv_writer = csv.writer(output_csv)
     #   csv_writer.writerow(['Distance to Plane'])
      #  csv_writer.writerows([[distance] for distance in distances_to_plane])

    #print(f"Distances to the plane saved to {output_csv_path}")

     plt.scatter(distances_to_plane, stg_values, marker='o', color='blue')
     plt.title('Deviation vs. Y Values')
     plt.xlabel('distance to plane')
     plt.ylabel('abc Values')
     plt.grid(True)
     plt.show()


if __name__ == "__main__":
    main()
```
# calculation of gap in partition coefficient
```
r=1.9872036*10**-3 # 
t=298.15
hartree2kcm=


g_oct=
g_water=
g_gap=(g_oct-g_water)*hartree2kcm
g_gap=-3.3944058230052323
log_p=(-g_gap)/(2.303*r*t)

print(log_p)
```
# merge csv files into one table with proper alignment 
```
import pandas as pd
import numpy as np

# Read CSV files
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Concatenate dataframes side by side
merged_df = pd.concat([df1, df2, df3], axis=1)

# Round off numbers to 3 decimal places
merged_df = merged_df.round(3)

# Display the resulting dataframe
Nrows=len(merged_df.values[0])+1
Ncols=len(merged_df.values)-1


for i in range(Nrows):
    print(i+1,end=' & ')
    for j in range(Ncols-1):
        print(merged_df.values[i][j],end=' & ')
        if np.mod(j+1,3)==0:
            print(end='& ')
    for j in range(Ncols-1,Ncols):
        print(merged_df.values[i][j],end=' \\\\ ')
    print('')
```
# Print different types of error from csv files
```
import csv
import numpy as np

def read_column(filename, column_index):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                value = float(row[column_index])
                data.append(value)
            except (ValueError, IndexError):
                pass
    return data

def calculate_mean_std_error(data1, data2):
    minE =np.min(data1 - data2)
    maxE =np.max(data1 - data2)
    mse = np.mean(data1 - data2)
    mae = np.mean(np.abs(data1 - data2))
    sde = np.std(data1 - data2)

    return minE,maxE,mse, mae, sde
```
# print the rows of csv file within a range in a column
```
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('file1.csv')

# Set the range for the 4th column (assuming it's zero-indexed)
min_value = 0.5
max_value = 0.8

# Filter rows based on the range of values in the 4th column
filtered_df = df[(df.iloc[:, 4] >= min_value) & (df.iloc[:, 4] <= max_value)]

# Print the filtered DataFrame
print(filtered_df)
```
# range within range
```
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('file1.csv')

# Set the range for the 4th column (assuming it's zero-indexed)
min_value_4th_col = 0.3
max_value_4th_col = 0.5

# Filter rows based on the range of values in the 4th column
filtered_df_4th_col = df[(df.iloc[:, 4] >= min_value_4th_col) & (df.iloc[:, 4] <= max_value_4th_col)]

# Set the range for the 1st column
min_value_1st_col = 3
max_value_1st_col = 4

# Further filter rows based on the range of values in the 1st column
filtered_df = filtered_df_4th_col[(filtered_df_4th_col.iloc[:, 1] >= min_value_1st_col) & (filtered_df_4th_col.iloc[:, 1] <= max_value_1st_col)]

# Print the final filtered DataFrame
print(filtered_df)
```
# scattered plot of 2 csv files 
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file
df1 = pd.read_csv('file1.csv')

# Read the second CSV file
df2 = pd.read_csv('file2.csv')

# Extract the second column from each DataFrame
x_values = df2.iloc[:, 1]  # Assuming the second column is indexed at 1
y_values = df1.iloc[:, 1]  # Assuming the second column is indexed at 1

# Plotting
plt.scatter(x_values, y_values)
plt.xlabel('Second column of file2')
plt.ylabel('Second column of file1')
plt.title('Scatter Plot')
plt.show()
```
# How to find which rows has an empty 4th column in csv file
```
import pandas as pd

def find_rows_with_empty_fourth_column(csv_file):
    df = pd.read_csv(csv_file)
    empty_rows = df[df.iloc[:, 3].isnull()]
    return empty_rows

if __name__ == "__main__":
    csv_file = 'your_file.csv'  # Replace 'your_file.csv' with your CSV file path
    empty_rows = find_rows_with_empty_fourth_column(csv_file)
    if not empty_rows.empty:
        print("Rows with an empty fourth column:")
        print(empty_rows)
    else:
        print("No rows with an empty fourth column found.")
```
# to read a csv file and print the 12 rows having smallest values of 4th column
```
import pandas as pd

def print_smallest_12_rows(csv_file):
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Sort DataFrame based on the fourth column
    sorted_df = df.sort_values(by=df.columns[3])  # Assuming the fourth column is the last column
    
    # Print the first 12 rows (smallest values)
    print("First 12 rows with smallest values of the fourth column:")
    print(sorted_df.head(12))

if __name__ == "__main__":
    csv_file = 'your_file.csv'  # Replace 'your_file.csv' with the path to your CSV file
    print_smallest_12_rows(csv_file)
```
# write a program to plot kde for 4 different pairs of csv files and then plot in 2x2 panel plot
```
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Read CSV files and extract second column for each pair
data_pairs = []
for i in range(4):
    df1 = pd.read_csv(f'file1_{i+1}.csv')
    df2 = pd.read_csv(f'file2_{i+1}.csv')
    data_pairs.append((df1.iloc[:, 1], df2.iloc[:, 1]))

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

# Plot KDE for each pair in a subplot
for i, (data1, data2) in enumerate(data_pairs):
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    x_values = sorted(data1)
    
    row = i // 2
    col = i % 2
    axs[row, col].plot(x_values, kde1(x_values), label='KDE1')
    axs[row, col].plot(x_values, kde2(x_values), label='KDE2')
    axs[row, col].set_xlabel('X-axis label')  # Replace 'X-axis label' with your desired label
    axs[row, col].set_ylabel('Density')  # Replace 'Density' with your desired label
    axs[row, col].set_title(f'Pair {i+1}')  # Set title for each subplot
    axs[row, col].legend()

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
```
# read 2 csv files and plot the 2nd column of one file in y axis vs the second column of the other in x axis
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file
df1 = pd.read_csv('file1.csv')

# Read the second CSV file
df2 = pd.read_csv('file2.csv')

# Extract the second column from each DataFrame
x_values = df2.iloc[:, 1]  # Assuming the second column is indexed at 1
y_values = df1.iloc[:, 1]  # Assuming the second column is indexed at 1

# Plotting
plt.scatter(x_values, y_values)
plt.xlabel('Second column of file2')
plt.ylabel('Second column of file1')
plt.title('Scatter Plot')
plt.show()
```
# read 2 csv files and print the rows having the 5th column in a common range 
```
import csv

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

def filter_rows_by_range(data, column_index, min_value, max_value):
    filtered_rows = []
    for row in data:
        if min_value <= float(row[column_index]) <= max_value:
            filtered_rows.append(row)
    return filtered_rows

def main():
    file1_path = 'file1.csv'
    file2_path = 'file2.csv'
    
    # Read CSV files
    file1_data = read_csv(file1_path)
    file2_data = read_csv(file2_path)
    
    # Set range for the 5th column
    column_index = 4  # Assuming 5th column (0-indexed)
    min_value = 10
    max_value = 20
    
    # Filter rows by range for both files
    file1_filtered_rows = filter_rows_by_range(file1_data, column_index, min_value, max_value)
    file2_filtered_rows = filter_rows_by_range(file2_data, column_index, min_value, max_value)
    
    # Find common rows
    common_rows = []
    for row1 in file1_filtered_rows:
        for row2 in file2_filtered_rows:
            if row1[column_index] == row2[column_index]:
                common_rows.append(row1)
    
    # Print common rows
    print("Rows with values in common range in both files:")
    for row in common_rows:
        print(row)

if __name__ == "__main__":
    main()
```
# write a program to read a csv file which has the first column as the names of xyz files.read the xyz file of that same name and then print the following into a new csv file. The sixth column of the csv file, total number of atoms in the coordinate file, individual atoms in one column, the xyz coordinates of each atom in each column, 2nd ,third and 4th column of that csv files  all in different columns
```
import csv

def read_xyz_file(filename):
    atoms = []
    coordinates = []
    
    with open(filename, 'r') as f:
        iline=0
        
        if iline==0:
            num_atoms = int(f.readline())
            iline=iline+1
        
        for line in f:
            if iline>1:
                atom, x, y, z = line.split()
                atoms.append(atom)
                coordinates.append([float(x), float(y), float(z)])
            iline=iline+1
    return num_atoms, atoms, coordinates

def main():
    csv_filename = input("Enter CSV filename: ")
    xyz_files = []
    with open(csv_filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            if i < 3:  # Read only the first 3 rows
                xyz_files.append((row[0] + ".xyz", row[2], row[3], row[4]))  # Add required columns from original CSV

    output_rows = []
    for xyz_file, col3, col4, col5 in xyz_files:
        num_atoms, atoms, coordinates = read_xyz_file(xyz_file)
        output_rows.append([xyz_file, num_atoms, atoms, coordinates, col3, col4, col5])

    output_csv_filename = "output.csv"  # Output CSV filename
    with open(output_csv_filename, 'w', newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)
        csv_writer.writerow(["XYZ_File", "Num_Atoms", "Atoms", "Coordinates", "Column_3", "Column_4", "Column_5"])
        csv_writer.writerows(output_rows)

    print(f"Data written to {output_csv_filename}")

if __name__ == "__main__":
    main()
```
# read a csv file and find difference of 14th and 11th column for all rows
```
import csv

def calculate_difference(csv_filename):
    differences = []
    with open(csv_filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if present
        for row in csv_reader:
            if len(row) >= 14:  # Check if row has at least 14 columns
                try:
                    col_14 = float(row[13])  # Index 13 represents the 14th column (0-indexed)
                    col_11 = float(row[10])  # Index 10 represents the 11th column (0-indexed)
                    difference = col_14 - col_11
                    differences.append(difference)
                except ValueError:
                    print("Error: Non-numeric value found in columns 11 or 14 in a row.")
            else:
                print("Error: Row does not have enough columns.")

    return differences

def main():
    csv_filename = input("Enter the name of the CSV file: ")
    differences = calculate_difference(csv_filename)
    if differences:
        print("Differences between 14th and 11th columns for each row:")
        for difference in differences:
            print(difference)
    else:
        print("No differences calculated.")

if __name__ == "__main__":
    main()
```
# Find the union and intersection of 2 dataframes with pandas
```
import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({'A': [1, 2, 3, 4],
                    'B': ['a', 'b', 'c', 'd']})

df2 = pd.DataFrame({'A': [3, 4, 5, 6],
                    'B': ['c', 'd', 'e', 'f']})

# Union
union_df = pd.concat([df1, df2]).drop_duplicates()

# Intersection
intersection_df = pd.merge(df1, df2, how='inner')

print("Union:")
print(union_df)
print("\nIntersection:")
print(intersection_df)
```
# set a range for a data frame and print only those which satisfies it
```
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({'A': [10, 20, 30, 40, 50]})

# Define the range
lower_bound = 15
upper_bound = 35

# Filter the DataFrame based on the range
filtered_df = df[(df['A'] >= lower_bound) & (df['A'] <= upper_bound)]

# Print the filtered DataFrame
print(filtered_df)
```
# how to print unique entries in a list with frequencies
```
from collections import Counter

# Sample list
my_list = ['a', 'b', 'c', 'a', 'b', 'a', 'd', 'b', 'c']

# Count the frequency of each element in the list
frequency_counter = Counter(my_list)

# Print unique entries along with their frequencies
for item, frequency in frequency_counter.items():
    print(f"Item: {item}, Frequency: {frequency}")
```
# find the 10 largest values of one column in datafunction and the apply a condition of range for another column. print only those of the 10 which satisties it
```
import pandas as pd

# Example DataFrame
data = {'Column1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'Column2': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]}

df = pd.DataFrame(data)

# Find the 10 largest values of Column1
largest_values = df.nlargest(10, 'Column1')

# Apply condition to filter rows based on Column2
filtered_values = largest_values[largest_values['Column2'] >= 50]

# Print the filtered values
print(filtered_values)
```
# from the 2 data frames we found the smallest 25 values, now we want the corresponding index of those 25 values from the dataframe
```
import pandas as pd

# Assuming df1 and df2 are your DataFrames
# df1 and df2 should contain the same indices to get the correct corresponding indices

# Example DataFrames
df1 = pd.DataFrame({'A': [10, 20, 30, 40, 50],
                    'B': [1, 2, 3, 4, 5]})

df2 = pd.DataFrame({'A': [15, 25, 35, 45, 55],
                    'B': [6, 7, 8, 9, 10]})

# Concatenate the DataFrames
concatenated_df = pd.concat([df1, df2])

# Find the smallest 25 values
smallest_25_values = concatenated_df.min().nsmallest(25)

# Get the corresponding indices
corresponding_indices = concatenated_df[concatenated_df.isin(smallest_25_values)].stack().index.tolist()

print("Corresponding Indices:")
print(corresponding_indices)
```
# suppose i print 3 columns. the first two already have headers. How to include header in the 3rd one
```
# Assuming you have data for the first two columns and a header for the third column
column1_data = [1, 2, 3]
column2_data = ['A', 'B', 'C']
column3_header = "Header3"
column3_data = [10, 20, 30]

# Printing headers for the first two columns
print("Column1 Column2", column3_header)

# Printing data for all three columns
for data1, data2, data3 in zip(column1_data, column2_data, column3_data):
    print(data1, data2, data3)
```
# np function to arrange in ascending order
```
import numpy as np

# Example array
arr = np.array([3, 1, 5, 2, 4])

# Sort the array in ascending order
arr_sorted = np.sort(arr)

print("Sorted array:", arr_sorted)
```
# find the 10 largest values of one column in datafunction and the apply a condition of range for another column. print only those of the 10 which satisties it
```
import pandas as pd

# Example DataFrame
data = {'Column1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'Column2': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]}

df = pd.DataFrame(data)

# Find the 10 largest values of Column1
largest_values = df.nlargest(10, 'Column1')

# Apply condition to filter rows based on Column2
filtered_values = largest_values[largest_values['Column2'] >= 50]

# Print the filtered values
print(filtered_values)
```
# write a python program to read a xyz file and calculate the distance between the coordinates of the 6th atom and all the atoms
```
import math

def read_xyz_file(file_path):
    atoms = []
    with open(file_path, 'r') as file:
        num_atoms = int(file.readline())
        file.readline()  # Skip the comment line

        for _ in range(num_atoms):
            line = file.readline().split()
            atom_symbol, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
            atoms.append((atom_symbol, (x, y, z)))

    return atoms

def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def main():
    file_path = 'your_xyz_file.xyz'  # Replace with the path to your XYZ file
    atoms = read_xyz_file(file_path)

    if len(atoms) >= 6:
        sixth_atom = atoms[5]  # 0-based index, so the 6th atom is at index 5
        print(f"Coordinates of the 6th atom: {sixth_atom[1]}")

        for i, atom in enumerate(atoms):
            if i != 5:  # Skip the 6th atom
                distance = calculate_distance(sixth_atom, atom)
                print(f"Distance between the 6th atom and atom {i + 1}: {distance:.3f}")

    else:
        print("Not enough atoms in the file.")

if __name__ == "__main__":
    main()
```
# read 2 csv files and plot the 2nd column of one file in y axis vs the second column of the other in x axis
```
import pandas as pd
import matplotlib.pyplot as plt

 
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

x_values = df2.iloc[:, 1]  # Assuming the second column is indexed at 1
y_values = df1.iloc[:, 1]  # Assuming the second column is indexed at 1

plt.scatter(x_values, y_values)
plt.xlabel('Second column of file2')
plt.ylabel('Second column of file1')
plt.title('Scatter Plot')
plt.show()
```
# read 3 csv files and convert them into one table and round off the numbers to 3 decimal places and the gap between the numbers should be properly adjusted 
```
import pandas as pd

# Read CSV files
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Combine dataframes into one
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Round off numbers to 3 decimal places
merged_df = merged_df.round(3)

# Display the resulting dataframe
print(merged_df)
```
# Get the 10 with largest values of one datafrae and with in a range for for another
```
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({'A': [10, 20, 30, 40, 50],
                   'B': [5, 15, 25, 35, 45]})

# Define the range for column 'B'
range_B_lower = 20
range_B_upper = 40

# Filter the DataFrame based on the range for column 'B'
filtered_df = df[(df['B'] >= range_B_lower) & (df['B'] <= range_B_upper)]

# Get the 10 entries with largest values in column 'A' from the filtered DataFrame
top_10_largest_A = filtered_df.nlargest(10, 'A')

# Print the result
print(top_10_largest_A)
```
# plot an equation vs another
```
import numpy as np
import matplotlib.pyplot as plt

# Define the range for x1 and x2
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)

# Create a meshgrid for x1 and x2
X1, X2 = np.meshgrid(x1, x2)

# Define the equations
equation1 = X1**2 + X2**2 - 1
equation2 = 2*X1**2 - X2 - 1

# Plot the equations
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, equation1, levels=[0], colors='blue', label='x1^2 + x2^2 - 1 = 0')
plt.contour(X1, X2, equation2, levels=[0], colors='red', label='2*x1^2 - x2 - 1 = 0')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Intersection of Equations')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```
# prepinp.sh 
```
import os

Nmol = 100
geomfile='100_geom.xyz'

filedir = os.getcwd()

geom_file = open(geomfile, 'r')

for imol in range(Nmol):

    line = geom_file.readline().strip()

    if line:

        Nat = int(line)
        title = geom_file.readline().strip()

        mol=new_name = "Mol_{:05d}".format(imol+1)

        print(mol)

        geomfile='geom_DFT_S0.xyz'

        inputfile= open(geomfile, 'w')

        inputfile.write(f'{Nat}\n')
        inputfile.write(f'{mol}\n')

        for iat in range(1, Nat + 1):
            line = geom_file.readline().split()
            sym=line[0]
            R=[float(line[1]), float(line[2]), float(line[3])]
            inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

        inputfile.close()

        os.mkdir(os.path.join(filedir, mol))

        os.system(f'cp tddft.com {geomfile} {mol}/')

geom_file.close()
```
# find dis and sort
```
import math

def read_xyz_file(file_path):
    atoms = []
    with open(file_path, 'r') as file:
        num_atoms = int(file.readline())
        file.readline()  # Skip the comment line

        for _ in range(num_atoms):
            line = file.readline().split()
            atom_symbol, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
            atoms.append((atom_symbol, (x, y, z)))

    return atoms

def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1]
    x2, y2, z2 = atom2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def main():
    file_path = 'your_xyz_file.xyz'  # Replace with the path to your XYZ file
    atoms = read_xyz_file(file_path)

    if len(atoms) >= 6:
        sixth_atom = atoms[5]  # 0-based index, so the 6th atom is at index 5
        print(f"Coordinates of the 6th atom: {sixth_atom[1]}")

        distances = []
        for i, atom in enumerate(atoms):
            if i != 5:  # Skip the 6th atom
                distance = calculate_distance(sixth_atom, atom)
                distances.append((i + 1, distance))  # Atom index and distance

        # Sort distances in ascending order
        sorted_distances = sorted(distances, key=lambda x: x[1])

        # Print sorted distances and corresponding atoms
        for atom_index, distance in sorted_distances:
            atom_symbol = atoms[atom_index - 1][0]  # 0-based index adjustment
            print(f"Atom {atom_index} ({atom_symbol}): Distance = {distance:.3f}")

    else:
        print("Not enough atoms in the file.")

if __name__ == "__main__":
    main()
```
# write a program to copy one particular xyz per folder to a separate folder.
```
import os
import shutil

def copy_xyz_file(src_folder, dest_folder, filename='xyz_file_to_copy.xyz'):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file == filename:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)
                shutil.copyfile(src_path, dest_path)
                print(f"Copied {file} from {src_folder} to {dest_folder}")

def main():
    # Source folder containing subfolders with XYZ files
    src_root_folder = 'source_root_folder'
    
    # Destination folder where selected XYZ files will be copied
    dest_folder = 'destination_folder'

    # Filename of the XYZ file to copy
    xyz_filename = 'xyz_file_to_copy.xyz'

    # Create the destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Copy the specified XYZ file from each folder to the destination folder
    for root, dirs, files in os.walk(src_root_folder):
        copy_xyz_file(root, dest_folder, xyz_filename)

if __name__ == "__main__":
    main()
```
# Write a python program to create a csv file with the second and third column of the bash extract output
# 
```import os
import csv

def check_xyz_file(file_path):
    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [line.strip().split() for line in lines[2:] if len(line.strip().split()) == 4]
        for i in range(min(10, len(data))):
            atom_type = data[i][0]
            if atom_type == 'S':
                results.extend([1, 0, 0])
            elif atom_type == 'P':
                results.extend([0, 1, 0])
            elif atom_type == 'C':
                results.extend([0, 0, 1])
            else:
                results.extend([0, 0, 0])
    return results

def main():
    # Root directory containing folders with XYZ files
    root_dir = 'path_to_root_directory'

    # CSV file to store results
    csv_file = 'results.csv'

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Iterate through each folder and process XYZ files
        for foldername in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, foldername)
            if os.path.isdir(folder_path):
                xyz_file_path = os.path.join(folder_path, 'your_specific_xyz_file.xyz')
                if os.path.isfile(xyz_file_path):
                    results = check_xyz_file(xyz_file_path)
                    writer.writerow(results)
                    print(f"Processed {xyz_file_path}")

if __name__ == "__main__":
    main()
```
# write a python program to find the unknown vecytor X where there is AX=B, where A is a mxn coeffecient matrix, X is nx1 matrix and B is mx1 matrix. Here A is one csv file, and B is the first column of another csv file. the first column of b csv file is string. do accordingly
```
import numpy as np
import pandas as pd

# Load the coefficient matrix A from CSV
A = pd.read_csv('A.csv', header=None).to_numpy()

# Load the B matrix from CSV
B_data = pd.read_csv('B.csv', header=None)
B = B_data.iloc[:, 1:].to_numpy()

# Convert the string column in B to numeric
B_strings = B_data.iloc[:, 0]
B_numeric = pd.to_numeric(B_strings, errors='coerce').fillna(0).to_numpy()

# Perform the least squares calculation to find X
X, residuals, rank, singular_values = np.linalg.lstsq(A, B_numeric, rcond=None)

# Print the unknown vector X
print("The unknown vector X:")
print(X)
```
# 
```
def main():
    # Load coefficients matrix A from CSV
    A_filename = 'coefficients.csv'
    A = load_csv(A_filename)

    # Load vector B from CSV
    B_filename = 'vector_b.csv'
    B = load_csv(B_filename)

    # Check dimensions
    m_A, n_A = A.shape
    m_B, n_B = B.shape
    if m_A != m_B:
        print("Error: Number of rows in A does not match the length of B.")
        return
    if n_B != 1:
        print("Error: Vector B should have only one column.")
        return

    # Solve the linear equation AX = B
    X = solve_linear_equation(A, B)

    print("Solution vector X:")
    print(X)
```
```
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('your_file.csv')

# Print the second column
print(df.iloc[:, 1])
```
# I want to extract the 3rd column of all the csv files and put those columns side by side in a new csv file with the csv file name as the header of the columns
```
import os
import csv

def extract_third_columns(input_folder, output_file):
    # Get a list of CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    # Dictionary to store third columns from each CSV file
    third_columns = {}

    # Iterate through each CSV file
    for csv_file in csv_files:
        with open(os.path.join(input_folder, csv_file), 'r', newline='') as file:
            reader = csv.reader(file)
            # Extract the third column and store it in the dictionary
            third_columns[csv_file] = [row[2] for row in reader]

    # Write the collected third columns to the output CSV file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write headers
        writer.writerow([os.path.splitext(csv_file)[0] for csv_file in csv_files])
        # Transpose the data and write it to the output file
        for i in range(len(third_columns[csv_files[0]])):
            writer.writerow([third_columns[csv_file][i] for csv_file in csv_files])

    print(f"Third columns from {len(csv_files)} CSV files have been extracted and saved to {output_file}.")

# Example usage:
input_folder = 'input_folder_path'  # Replace 'input_folder_path' with the path to your folder containing CSV files
output_file = 'output_file.csv'     # Specify the name of the output CSV file
extract_third_columns(input_folder, output_file)
```
# plot for X1/4 + X2/8 + 11/8   and   -X1/8 - X2/4 + 1
```
import numpy as np
import matplotlib.pyplot as plt

# Define the range for x1 and x2
x1 = np.linspace(-10, 10, 400)
x2 = np.linspace(-10, 10, 400)

# Create a meshgrid for x1 and x2
X1, X2 = np.meshgrid(x1, x2)

# Define the equations
equation1 = X1/4 + X2/8 + 11/8
equation2 = -X1/8 - X2/4 + 1

# Plot the equations
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, equation1, levels=[0], colors='blue', label='X1/4 + X2/8 + 11/8 = 0')
plt.contour(X1, X2, equation2, levels=[0], colors='red', label='-X1/8 - X2/4 + 1 = 0')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Intersection of Equations')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```
# load 2 csv files. Then plot in y axis the 2nd column of first csv file - second column of the other. In x axis it will be the third column of the first csv file
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

# Extracting relevant columns
x = df1.iloc[:, 2]  # Third column of the first CSV file
y = df1.iloc[:, 1] - df2.iloc[:, 1]  # Second column of first CSV - Second column of second CSV

# Plotting
plt.plot(x, y)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Plotting Difference of Columns')
plt.grid(True)
plt.show()
```
# Here also add something to print the index of the points beside the points in the plot
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
df1 = pd.read_csv('a1.csv')
df2 = pd.read_csv('a1.csv')

# Extracting relevant columns
x = df1.iloc[:, 2]  # Third column of the first CSV file
y = df1.iloc[:, 1] - df2.iloc[:, 1]  # Second column of first CSV - Second column of second CSV
y = np.abs(y)

# Plotting
plt.scatter(x, y)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Plotting Difference of Columns')
plt.grid(True)

# Annotating points with index
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, str(i), fontsize=8, verticalalignment='bottom', horizontalalignment='right')

plt.show()
```
# a python program to change the 3rd, 4th and fifth column of one csv file with the 3rd, 4th column of the new csv file
```
import csv

def modify_csv(input_file, new_file):
    # Read the contents of the original CSV file
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Read the contents of the new CSV file
    with open(new_file, 'r', newline='') as f:
        reader = csv.reader(f)
        new_data = list(reader)

    # Modify the specified columns in the original CSV file with corresponding columns from the new CSV file
    for i, row in enumerate(data):
        if i < len(new_data):
            data[i][2] = new_data[i][2]  # Modify 3rd column
            data[i][3] = new_data[i][3]  # Modify 4th column
            data[i][4] = new_data[i][3]  # Modify 5th column

    # Write the modified data to a new CSV file
    with open('modified.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

if __name__ == "__main__":
    input_file = 'original.csv'  # Specify the original CSV file
    new_file = 'new.csv'  # Specify the new CSV file
    modify_csv(input_file, new_file)
```
# find column 6-column 5 in the csv file and arrange the results in ascending order

```
import csv

def print_sorted_differences(input_file):
    # Read the contents of the CSV file
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        differences = []

        # Calculate the difference between column 6 and column 5 for each row
        for row in reader:
            col_5 = float(row[5])
            col_6 = float(row[6])
            difference = col_6 - col_5
            differences.append(difference)

        # Sort and print the differences in ascending order
        sorted_differences = sorted(differences)
        for diff in sorted_differences:
            print(diff)

if __name__ == "__main__":
    input_file = 'original.csv'  # Specify the CSV file
    print_sorted_differences(input_file)
```
```
import os
import csv

def collect_last_columns(input_folder, output_file):
    # Get a list of CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    # Dictionary to store data from each CSV file
    csv_data = {}

    # Iterate through each CSV file
    for csv_file in csv_files:
        with open(os.path.join(input_folder, csv_file), 'r', newline='') as file:
            reader = csv.reader(file)
            # Read each row of the CSV file and store it in the dictionary
            csv_data[csv_file] = [row for row in reader]

    # Write the collected data to the output CSV file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write headers
        writer.writerow(['File Name'] + [f'Column_{i+1}' for i in range(len(csv_data[csv_files[0]][0]))])
        # Write data for each CSV file
        for csv_file in csv_files:
            for row in csv_data[csv_file]:
                writer.writerow([csv_file] + row)

    print(f"All columns from {len(csv_files)} CSV files have been collected and saved to {output_file}.")

# Example usage:
input_folder = 'input_folder_path'  # Replace 'input_folder_path' with the path to your folder containing CSV files
output_file = 'output_file.csv'     # Specify the name of the output CSV file
collect_last_columns(input_folder, output_file)
```
```
import csv

def print_sorted_differences(input_file):
    # Read the contents of the CSV file
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        differences = []

        # Calculate the difference between column 6 and column 5 for each row
        for row in reader:
            col_5 = float(row[5])
            col_6 = float(row[6])
            difference = col_6 - col_5
            differences.append(difference)

        # Sort and print the differences in ascending order
        sorted_differences = sorted(differences)
        for diff in sorted_differences:
            print(diff)

if __name__ == "__main__":
    input_file = 'original.csv'  # Specify the CSV file
    print_sorted_differences(input_file)
```
# Histogram
```
import matplotlib.pyplot as plt
import pandas as pd

def draw_histogram(csv_file, column1, column2):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    
    # Extract the columns
    col1_data = data[column1]
    col2_data = data[column2]
    
    # Create a figure and axis object
    fig, ax = plt.subplots()
    
    # Plot histograms for both columns
    ax.hist(col1_data, alpha=0.5, label=column1)
    ax.hist(col2_data, alpha=0.5, label=column2)
    
    # Add labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of {} and {}'.format(column1, column2))
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

# Example usage
csv_file = 'data.csv'  # Replace 'data.csv' with your CSV file
column1 = 'Column1'     # Replace 'Column1' with the name of the first column
column2 = 'Column2'     # Replace 'Column2' with the name of the second column

draw_histogram(csv_file, column1, column2)
```
# white a python program to convert a text file to csv file format
```
import csv

def convert_text_to_csv(input_file, output_file, delimiter=','):
    # Open the text file for reading
    with open(input_file, 'r') as infile:
        # Open the CSV file for writing
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=delimiter)
            
            # Read each line from the text file
            for line in infile:
                # Split the line using the delimiter
                data = line.strip().split(delimiter)
                
                # Write the data to the CSV file
                writer.writerow(data)

# Example usage
input_file = 'input.txt'   # Replace 'input.txt' with the name of your text file
output_file = 'output.csv' # Name of the output CSV file

convert_text_to_csv(input_file, output_file)
```
# bar plot
```
import matplotlib.pyplot as plt
import pandas as pd

def plot_bar_chart(csv_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file, header=None, names=['Category', 'Value'])
    
    # Extracting category labels and corresponding values
    categories = data['Category']
    values = data['Value']
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Adjust figure size if necessary
    plt.bar(categories, values, color='skyblue')
    
    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Plot')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Example usage
csv_file = 'data.csv'  # Replace 'data.csv' with the name of your CSV file
plot_bar_chart(csv_file)
```
# write a python program to find the unknown vector X where there is AX=B, where A is a mxn coeffecient matrix, X is nx1 matrix and B is mx1 matrix. Here A is one csv file, and B is the first column of another csv file. the first column of b csv file is string. 
```
import numpy as np
import pandas as pd

# Load the coefficient matrix A from CSV
A = pd.read_csv('A.csv', header=None).to_numpy()

# Load the B matrix from CSV
B_data = pd.read_csv('B.csv', header=None)
B = B_data.iloc[:, 1:].to_numpy()

# Convert the string column in B to numeric
B_strings = B_data.iloc[:, 0]
B_numeric = pd.to_numeric(B_strings, errors='coerce').fillna(0).to_numpy()

# Perform the least squares calculation to find X
X, residuals, rank, singular_values = np.linalg.lstsq(A, B_numeric, rcond=None)

# Print the unknown vector X
print("The unknown vector X:")
print(X)
```
# find the lowest value of the 2nd column of a csv file, and then find its corresponding 3rd column value. Now multiply the second column value by 27, then find the difference between that value and the corresponding 3rd column value found
```
import csv

# Function to find the lowest value in the second column and corresponding third column value
def find_min_and_corresponding(csv_file):
    min_value = float('inf')
    corresponding_value = None
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3:
                second_column_value = float(row[1])
                if second_column_value < min_value:
                    min_value = second_column_value
                    corresponding_value = float(row[2])
    return min_value, corresponding_value

# Calculate the difference between the multiplied second column value and corresponding third column value
def calculate_difference(min_value, corresponding_value):
    multiplied_value = min_value * 27
    difference = multiplied_value - corresponding_value
    return difference

# Main function
def main():
    csv_file = "your_csv_file.csv"  # Replace "your_csv_file.csv" with your actual CSV file path
    min_value, corresponding_value = find_min_and_corresponding(csv_file)
    if min_value is not None and corresponding_value is not None:
        difference = calculate_difference(min_value, corresponding_value)
        print("Lowest value in the second column:", min_value)
        print("Corresponding value in the third column:", corresponding_value)
        print("Difference after multiplication:", difference)
    else:
        print("CSV file is empty or doesn't contain necessary columns.")

if __name__ == "__main__":
    main()
```
# summation of a column of csv file using numpy
```
import numpy as np
import csv

def sum_second_column(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        data = [float(row[1]) for row in reader]

    total_sum = np.sum(data)
    return total_sum

# Example usage:
csv_file = 'output.csv'  # Replace with the path to your CSV file
total_sum = sum_second_column(csv_file)
print("Summation of the second column:", total_sum)
```

# Plot columns with respect to index
```
import matplotlib.pyplot as plt
import pandas as pd

# Load data from the first CSV file
df1 = pd.read_csv('file1.csv')
x1 = df1.index
y1_col2 = df1.iloc[:, 1]  # 2nd column
y1_col3 = df1.iloc[:, 2]  # 3rd column

# Load data from the second CSV file
df2 = pd.read_csv('file2.csv')
x2 = df2.index
y2_col2 = df2.iloc[:, 1]  # 2nd column
y2_col3 = df2.iloc[:, 2]  # 3rd column

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x1, y1_col2, label='File 1 - Column 2')
plt.plot(x1, y1_col3, label='File 1 - Column 3')
plt.plot(x2, y2_col2, label='File 2 - Column 2')
plt.plot(x2, y2_col3, label='File 2 - Column 3')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Spectrum Plot')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
```
# convert .spectrum file to .csv file
```
import csv

def convert_spectrum_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Assuming .spectrum file format: wavelength intensity error flag
    data = [line.strip().split() for line in lines]

    # Write data to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    print(f"Conversion complete. CSV file saved as {output_file}")

# Example usage:
input_file = 'input.spectrum'  # Replace 'input.spectrum' with your file path
output_file = 'output.csv'  # Specify the name/path for the output CSV file
convert_spectrum_to_csv(input_file, output_file)
```
# save the plot as pdf
```
import matplotlib.pyplot as plt
import pandas as pd

# Load data from the first CSV file
df1 = pd.read_csv('file1.csv')
x1 = df1.iloc[:, 0]  # Assuming the index is the first column
y1_col2 = df1.iloc[:, 1]  # 2nd column
y1_col3 = df1.iloc[:, 2]  # 3rd column

# Load data from the second CSV file
df2 = pd.read_csv('file2.csv')
x2 = df2.iloc[:, 0]  # Assuming the index is the first column
y2_col2_scaled = df2.iloc[:, 1] / 240  # 2nd column scaled
y2_col3_scaled = df2.iloc[:, 2] / 240  # 3rd column scaled

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x1, y1_col2, label='File 1 - Column 2')
plt.plot(x1, y1_col3, label='File 1 - Column 3')
plt.plot(x2, y2_col2_scaled, label='File 2 - Column 2 (Scaled)')
plt.plot(x2, y2_col3_scaled, label='File 2 - Column 3 (Scaled)')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Spectrum Plot')
plt.legend()

# Save plot as PDF
plt.grid(True)
plt.savefig('spectrum_plot.pdf')

# Show plot
plt.show()
```
# Learn to plot Jablonski diagram withpython
# calculation with energy and propert using Boltzman constant
```
import pandas as pd
import numpy as np

def boltzmann_average(E, unit, prop, T):
  
    # https://en.wikipedia.org/wiki/Boltzmann_constant 
    if ( unit == 'eV' ):
        kB=8.617333262*10**-5 # eV/K
    elif ( unit == 'kcm' ):
        kB=1.987204259*10**-3 # kcal/mol/K
    elif ( unit == 'kjm' ):
        kB=3.166811563*10**-6 # hartree/K
    elif ( unit == 'hartree' ):
        kB=8.314462618*10**-3 # kJ/mol/K
    elif ( unit == 'cmi' ):
        kB=0.695034800        # cmi/K

    # Convert energy to Boltzmann factor
    boltzmann_factors = np.exp(-E / (kB * T))

    # Calculate weighted average of property
    ave_prop = np.sum(prop * boltzmann_factors) / np.sum(boltzmann_factors)

    return ave_prop

# Read CSV file
df = pd.read_csv('prop.csv')

# Extract energy and property 
E = df['Energy']
prop = df['Property']

# Define the unit of energies
unit='cmi'

# Define temperature values for averaging
Ts = [28, 77, 100, 300, 400, 500, 1000, 2000, 3000]  # Add more temperatures as needed

# Perform Boltzmann averaging for each temperature
for T in Ts:
    ave_prop = boltzmann_average(E, unit,prop, T)
    print(f'Temperature: {T} K, Boltzmann-Averaged Property: {ave_prop:.4f}')
print(f'Fast thermalization average is:                  {np.mean(prop):.4f}')
```
# find the lowest value of the 2nd column of a csv file, and then find its corresponding 3rd column value. Now multiply the second column value by 27, then find the difference between that value and the corresponding 3rd column value found
```
import csv

# Function to find the lowest value in the second column and corresponding third column value
def find_min_and_corresponding(csv_file):
    min_value = float('inf')
    corresponding_value = None
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3:
                second_column_value = float(row[1])
                if second_column_value < min_value:
                    min_value = second_column_value
                    corresponding_value = float(row[2])
    return min_value, corresponding_value

# Calculate the difference between the multiplied second column value and corresponding third column value
def calculate_difference(min_value, corresponding_value):
    multiplied_value = min_value * 27
    difference = multiplied_value - corresponding_value
    return difference

# Main function
def main():
    csv_file = "your_csv_file.csv"  # Replace "your_csv_file.csv" with your actual CSV file path
    min_value, corresponding_value = find_min_and_corresponding(csv_file)
    if min_value is not None and corresponding_value is not None:
        difference = calculate_difference(min_value, corresponding_value)
        print("Lowest value in the second column:", min_value)
        print("Corresponding value in the third column:", corresponding_value)
        print("Difference after multiplication:", difference)
    else:
        print("CSV file is empty or doesn't contain necessary columns.")

if __name__ == "__main__":
    main()
```
# To  draw horizontal lines for jablonski
```
import matplotlib.pyplot as plt

# Function to draw a bunch of horizontal lines
def draw_horizontal_lines(y_positions, x_shift=0, color='black'):
    for y in y_positions:
        plt.axhline(y=y, xmin=x_shift, xmax=1, color=color)

# Define the positions of the first bunch of lines
lower_lines = [1, 2, 3, 4]

# Define the positions of the second bunch of lines (shifted a little to the right)
upper_lines = [6, 7, 8, 9]

# Define the amount of shift for the second bunch of lines
shift_amount = 0.5

# Draw the lines
draw_horizontal_lines(lower_lines)
draw_horizontal_lines(upper_lines, x_shift=shift_amount)

# Adjust the x-axis limits to give space for the second bunch of lines
plt.xlim(0, 1 + shift_amount)

# Show the plot
plt.show()
```
# Plot histograms with the 2nd column of each of the 3 csv files. One histogram should be in red, one in blue, one in green. 
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Plot histograms
plt.hist(df1.iloc[:, 1], color='red', alpha=0.5, label='File 1')
plt.hist(df2.iloc[:, 1], color='blue', alpha=0.5, label='File 2')
plt.hist(df3.iloc[:, 1], color='green', alpha=0.5, label='File 3')

# Add labels and legend
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Second Column')
plt.legend()

# Show plot
plt.show()
```
# Plot histograms with the 2nd column of each of the 3 csv files. One histogram should be in red, one in blue, one in green. 
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Plot histograms
plt.hist(df1.iloc[:, 1], color='red', alpha=0.5, label='File 1')
plt.hist(df2.iloc[:, 1], color='blue', alpha=0.5, label='File 2')
plt.hist(df3.iloc[:, 1], color='green', alpha=0.5, label='File 3')

# Add labels and legend
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Second Column')
plt.legend()

# Show plot
plt.show()
```
## Remember to use header=NONE
# 6-panel histogram plot, colour light , bin width same
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'

df1 = pd.read_csv(file1, header=None)
df2 = pd.read_csv(file2, header=None)
df3 = pd.read_csv(file3, header=None)

# Define the columns to plot
columns_to_plot = [1, 3, 4, 5, 6, 7]  # 2nd, 4th, 5th, 6th, 7th, and 8th columns

# Define titles for each subplot
titles = ['2nd Column', '4th Column', '5th Column', '6th Column', '7th Column', '8th Column']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Plot histograms for each column
for i, ax in enumerate(axes.flat):
    col_index = columns_to_plot[i]
    min_val = min(df1.iloc[:, col_index].min(), df2.iloc[:, col_index].min(), df3.iloc[:, col_index].min())
    max_val = max(df1.iloc[:, col_index].max(), df2.iloc[:, col_index].max(), df3.iloc[:, col_index].max())
    bin_width = (max_val - min_val) / 30  # Adjust the denominator for desired number of bins
    ax.hist(df1.iloc[:, col_index], bins=int((max_val - min_val) / bin_width), range=(min_val, max_val), color='red', alpha=0.3, label='File 1')
    ax.hist(df2.iloc[:, col_index], bins=int((max_val - min_val) / bin_width), range=(min_val, max_val), color='blue', alpha=0.3, label='File 2')
    ax.hist(df3.iloc[:, col_index], bins=int((max_val - min_val) / bin_width), range=(min_val, max_val), color='green', alpha=0.3, label='File 3')
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(titles[i])
    ax.legend()

# Adjust layout
plt.tight_layout()

# Save the figure as a PDF
plt.savefig('histograms.pdf')

# Show plot
plt.show()
```
# Morse potetial plot
```
import numpy as np
import matplotlib.pyplot as plt

# Define Morse potential function
def morse_potential(r, D, a, re):
    return D * (1 - np.exp(-a * (r - re)))**2

# Define parameters for Morse potential
D1 = 1.0    # Depth of potential well for first curve
a1 = 1.0    # Width parameter for first curve
re1 = 1.0   # Equilibrium bond length for first curve

D2 = 0.8    # Depth of potential well for second curve
a2 = 1.2    # Width parameter for second curve
re2 = 1.2   # Equilibrium bond length for second curve

# Generate r values
r = np.linspace(0.1, 5, 100)

# Calculate Morse potential values for first curve
potential1 = morse_potential(r, D1, a1, re1)

# Calculate Morse potential values for second curve
potential2 = morse_potential(r, D2, a2, re2)

# Plot Morse potential curve 1
plt.plot(r, potential1, label='Morse Potential 1')

# Plot Morse potential curve 2 shifted upwards
plt.plot(r, potential2 + max(potential1), label='Morse Potential 2')

# Define levels for horizontal lines (using the larger of the two depths)
levels = np.linspace(0, max(D1, D2), 5)

# Plot horizontal lines
for level in levels:
    plt.axhline(y=level, color='gray', linestyle='--', linewidth=0.5)

# Add labels and legend
plt.xlabel('Interatomic distance (r)')
plt.ylabel('Potential Energy')
plt.title('Morse Potential Curves')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
```
# Min E and Max E with mse mae sde errors

# plot
```
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Length of the lines
line_length = 0.2  # Adjust this value as needed

# Plotting with longer lines
for i in range(len(x)):
    plt.plot([x[i] - line_length/2, x[i] + line_length/2], [y[i], y[i]], color='blue')

# Customizing plot
plt.title('Plot with Small Horizontal Lines (Slightly Longer)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# Display the plot
plt.show()
```
# compare sdf and xyz file with ConnG0
```
! compile  gfortran CheckConnGO.f90 -o CheckConnGO.x
program conngo

  implicit none

  integer                         :: i, tmpi
  integer                         :: Nat, Nconn, igeo, nlong1, nlong2
  integer                         :: iat, iconn
  integer, allocatable            :: conn(:,:)

  character(len=500)              :: arg, cmd
  character(len=500)              :: title1, title2, title, title3, gaussinp
  character(len=500)              :: file1, file2, file3

  character(len=2), allocatable   :: sy(:)
  character(len=200), allocatable :: tit1(:), tit2(:)

  double precision                :: RR1(1:3), RR2(1:3), R12, dR(1:3), MSD, MPAD, MaxAD, long1, long2
  double precision, allocatable   :: R1(:,:), R2(:,:), dist1(:), dist2(:), distsort1(:), distsort2(:)

  character(len=10)               :: ls1, ls2

  double precision, parameter     :: rthresh = 1.68d0 ! Threshold to detect ultralong bonds 

  call getarg(1, arg)
  file1=trim(arg)

  call getarg(2, arg)
  file2=trim(arg)

  call getarg(3, arg)
  file3=trim(arg)

  !=== Read SDF (file-1)
  open(unit=101, file=trim(file1))
  read(101,*)title1
  read(101,*)title2
  read(101,*)
  read(101,'(2i3,a)')Nat, Nconn, title3
  allocate( R1(1:Nat,1:3), R2(1:Nat,1:3), sy(1:Nat), tit1(1:Nat), tit2(1:Nconn), conn(1:Nconn,1:3) )
  allocate( dist1(1:Nconn), dist2(1:Nconn) )
  allocate( distsort1(1:Nconn), distsort2(1:Nconn) )
  !=== read xyz
  do iat = 1, Nat
    read(101,'(3f10.4,a)') R1(iat,1:3), tit1(iat)
  enddo
  !=== read connectivities
  do iconn = 1, Nconn
    read(101,'(3i3,a)') conn(iconn,1:3), tit2(iconn)
    RR1 = R1( conn(iconn,1), 1:3 )
    RR2 = R1( conn(iconn,2), 1:3 )
    dR = RR1 - RR2
    R12 = dsqrt(dot_product(dR,dR))
    !=== dist1 has distances corresponding to connectivities from file-1
    dist1(iconn) = R12
  enddo
  close(101)

  !=== Read XYZ file (file-2)
  open(unit=101, file=trim(file2))
  read(101,*) Nat
  read(101,*) title1
  do iat = 1, Nat
    read(101,*) sy(iat), R2(iat,1:3)
  enddo
  close(101)

  do iconn = 1, Nconn
    RR1 = R2( conn(iconn,1), 1:3 )
    RR2 = R2( conn(iconn,2), 1:3 )
    dR = RR1 - RR2
    R12 = dsqrt(dot_product(dR,dR))
    !=== dist2 has distances corresponding to connectivities from file-2
    dist2(iconn) = R12
  enddo

  !=== third file, make new sdf
  open(unit=101, file=trim(file3))
  write(101,'(a)')trim(title1)
  write(101,'(x,a)')trim(title2)
  write(101,*)
  write(101,'(2i3,a)')Nat, Nconn, trim(title3)
  !=== write coordinates from file-2
  do iat = 1, Nat
    write(101,'(3f10.4,x,a)') R2(iat,1:3), trim(tit1(iat))
  enddo
  !=== write connectivities from file-1
  do iconn = 1, Nconn
    write(101,'(3i3,a)') conn(iconn,1:3), trim(tit2(iconn))
  enddo
  write(101,'(a)')"M  END"
  write(101,'(a)')"$$$$"
  close(101)

  MSD = 0d0

  write(*,*)
  write(*,'(a)')"== connectivities"

  nlong1 = 0
  nlong2 = 0

  write(*,'(a)')"                     File-1                  File-2                Deviation"
  do iconn = 1, Nconn

    ls1 = '         '
    ls2 = '         '

    if ( dist1(iconn) > rthresh ) then
      ls1 = 'ultralong'
      nlong1 = nlong1 + 1
    endif
    if ( dist2(iconn) > rthresh ) then
      ls2 = 'ultralong'
      nlong2 = nlong2 + 1
    endif

    if ( conn(iconn,3) .eq. 1) then
      write(*,'(3i3,2x,3a,f10.4,2x,a,2x,f10.4,2x,a,2x,f10.4)') conn(iconn,1:3), sy(conn(iconn,1)), "- ", sy(conn(iconn,2)), &
      dist1(iconn), ls1, dist2(iconn), ls2, dist1(iconn)-dist2(iconn)
    elseif ( conn(iconn,3) .eq. 2) then
      write(*,'(3i3,2x,3a,f10.4,2x,a,2x,f10.4,2x,a,2x,f10.4)') conn(iconn,1:3), sy(conn(iconn,1)), "= ", sy(conn(iconn,2)), &
      dist1(iconn), ls1, dist2(iconn), ls2, dist1(iconn)-dist2(iconn)
    elseif ( conn(iconn,3) .eq. 3) then
      write(*,'(3i3,2x,3a,f10.4,2x,a,2x,f10.4,2x,a,2x,f10.4)') conn(iconn,1:3), sy(conn(iconn,1)), "# ", sy(conn(iconn,2)), &
      dist1(iconn), ls1, dist2(iconn), ls2, dist1(iconn)-dist2(iconn)
    endif
    MSD = MSD + (dist1(iconn)-dist2(iconn))**2
  enddo

  open(unit=101, file='scr')
  do iconn = 1, Nconn
    write(101,*) dist1(iconn)
  enddo
  close(101)

  write(cmd, '(a)') "sort -n -r scr > scr1; mv scr1 scr"
  call system(trim(cmd))

  open(unit=101, file='scr')
  do iconn = 1, Nconn
    read(101,*) distsort1(iconn)
  enddo
  close(101)

  write(*,*)
  write(*,'(a,i4)')"== bond lengths in file-1 in descending order, # ultralong bonds = ", nlong1
  do iconn = 1, Nconn
    write(*,'(f10.4)',advance='no') distsort1(iconn)
  enddo
  write(*,*)

  open(unit=101, file='scr')
  do iconn = 1, Nconn
    write(101,*) dist2(iconn)
  enddo
  close(101)
  write(*,*)

  write(cmd, '(a)') "sort -n -r scr > scr1; mv scr1 scr"
  call system(trim(cmd))

  open(unit=101, file='scr')
  do iconn = 1, Nconn
    read(101,*) distsort2(iconn)
  enddo
  close(101)

  write(*,'(a,i4)')"== bond lengths in file-2 in descending order, # ultralong bonds = ", nlong2
  do iconn = 1, Nconn
    write(*,'(f10.4)',advance='no') distsort2(iconn)
  enddo
  write(*,*)

  long1 = maxval(dist1)
  long2 = maxval(dist2)

  write(*,*)
  if ( (long1 .gt. 1.75d0) .and. (long2 .gt. 1.75d0) ) then
    write(*,'(a)') "** BAD order or BROKEN structure in both file-1 and file-2 **"
  elseif ( (long1 .gt. 1.75d0) .and. (long2 .le. 1.75d0) ) then
    write(*,'(a)') "** BAD order or BROKEN structure in file-1 **"
  elseif ( (long2 .gt. 1.75d0) .and. (long1 .le. 1.75d0) ) then
    write(*,'(a)') "** BAD order or BROKEN structure in file-2 **"
  elseif ( (long2 .le. 1.75d0) .and. (long1 .le. 1.75d0) ) then
    write(*,'(a)') "** Geometries in file-1 and file-2 seem alright, no broken structures detected **"
  endif

  MSD = sqrt(MSD/dfloat(Nconn))
  !=== Mean Percentage Absolute Deviation, w.r.t. the dist1 from file-1
  MPAD = sum(abs( (dist1-dist2)/dist1 ) * 100d0 )/ dfloat(Nconn)  
  !=== MaxAD from dist1 and dist2 stored above
  MaxAD = maxval(abs( dist1-dist2))

  write(*,*)
  write(*,'(a)')"== Mean square deviation of bond lengths from file-1 and file-2"
  write(*,'(x,a,f10.4,a)')"MSD  = ", MSD, " Ang"
  write(*,*)
  write(*,'(a)')"== Maximum absolute deviation in bond lengths from file-1 and file-2"
  write(*,'(x,a,f10.4,a)')"MaxAD= ", MaxAD, " Ang"
  write(*,*)
  write(*,'(a)')"== Mean percentage absolute deviation in bond lengths from file-1 and file-2"
  write(*,'(x,a,f10.4)')"MPAD = ", MPAD
  write(*,*)

  write(*,'(a)')"== Outcome of the Connectivity preserving Geometry Optimization"
  if ( (MPAD .lt. 5d0) .and. (MaxAD .lt. 0.2d0) )  then
    write(*,'(a)')"** ConnGO PASS [MPAD < 5, MaxAD < 0.2 Angstrom] **"
  else
    write(*,'(a)')"** ConnGO FAIL **"
  endif
  write(*,*)

  write(cmd, '(a)') "rm -f scr"
  call system(trim(cmd))

  deallocate(R1, R2, sy, tit1, tit2, conn, dist1, dist2, distsort1, distsort2)

end program conngo
```
# extract cartesian coordinates from output and create folders with xyz file present and copy an input file to all
```
import os
import bz2

# Function to extract geometries from trajectory file
def extract_geometries(file_path):
    geometries = []
    with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        geom_start = False
        for line in lines:
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                geom_start = True
                geometry = []
                next(file)  # Skip the header line
                next(file)  # Skip the line indicating the number of atoms
            elif geom_start:
                atom_info = line.split()[1:]
                if len(atom_info) > 0:  # Check if it's not an empty line
                    geometry.append(atom_info)
                    if len(geometry) == 20:  # Assuming 20 atoms per geometry
                        geometries.append(geometry)
                        geometry = []
        return geometries

# Function to save geometries into XYZ files
def save_geometries(geometries, output_folder):
    for i, geometry in enumerate(geometries):
        traj_folder = os.path.join(output_folder, f'traj{i+1}')
        os.makedirs(traj_folder, exist_ok=True)
        xyz_file_path = os.path.join(traj_folder, f'traj{i+1}.xyz')
        with open(xyz_file_path, 'w') as xyz_file:
            xyz_file.write(f"{len(geometry)}\n\n")
            for atom_info in geometry:
                xyz_file.write(f"{atom_info[0]} {' '.join(atom_info[1:])}\n")
        print(f"XYZ file saved: {xyz_file_path}")

# Main function
def main():
    # Path to the compressed trajectory file
    compressed_trajectory_file = "path/to/compressed_trajectory_file.out.bz2"

    # Output folder
    output_folder = "path/to/output_folder"

    # Extract geometries from the compressed trajectory file
    geometries = extract_geometries(compressed_trajectory_file)

    # Save geometries into XYZ files inside corresponding folders
    save_geometries(geometries, output_folder)

if __name__ == "__main__":
    main()
```
# merge csv into one csv
```
import csv

def merge_csv_files(file1, file2, file3, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(file3, 'r') as f3, open(output_file, 'w', newline='') as out_file:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        reader3 = csv.reader(f3)
        writer = csv.writer(out_file)

        # Skip headers
        next(reader1)
        next(reader2)
        next(reader3)

        for col1, col2, col3 in zip(reader1, reader2, reader3):
            merged_row = col1 + col2 + col3
            writer.writerow(merged_row)

# Usage example
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'
output_file = 'merged.csv'

merge_csv_files(file1, file2, file3, output_file)
```
# smiles combination
```
file1=open('invest.smi','w')
ag = ['c','n']
bg = ['c','n']
cg = ['c','n']
dg = ['c','n']
eg = ['c','n']
fg = ['c','n']
gg = ['c','n']
hg = ['c','n']
ig = ['c','n']
jg = ['c','n']
kg = ['c','n']
lg = ['c','n']

num = 0 
for a in ag:
    for b in bg:
        for c in cg:
            for d in dg:
                for e in eg:
                    for f in fg:
                        for g in gg:
                            for h in hg:
                                for i in ig:
                                    for j in jg:
                                        for k in kg:
                                            for l in lg:
                                                mol = "{}1{}{}2{}{}{}{}3{}{}{}{}({}1)N23".format(a,b,c,d,e,f,g,h,i,j,k,l) 
                                                name='invest_'+str(num)
                                                file1.write(mol+' '+name+'\n')
                                                num = num + 1
print(num)
```
# remove comma
```
import csv

def csv_to_xyz(input_file, output_file):
    with open(input_file, 'r') as csv_file, open(output_file, 'w') as xyz_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # Check if the row has at least two elements
            if len(row) >= 2:
                # Remove commas and second column
                row.pop(1)
                # Write the row to the xyz file with spaces between each column
                xyz_file.write(' '.join(row) + '\n')

# Example usage:
input_file = 'input.csv'
output_file = 'output.xyz'
csv_to_xyz(input_file, output_file)
```
# extrac single point energies from output file
```
import numpy as np
import csv

# Read lines from input file
with open('scan.out', 'r') as file:
    lines = file.readlines()

# List to store energies
energies = []

# Extract energies from FINAL SINGLE POINT ENERGY lines
for line in lines:
    if "FINAL SINGLE POINT ENERGY" in line:
        energy_str = line.split()[-1]  # Extract the energy value
        energies.append(float(energy_str))  # Convert energy to float and store

# Write energies to a CSV file
with open('energies.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Energy'])  # Write header
    writer.writerows(map(lambda x: [x], energies))

print("Energies saved to energies.csv")
```
# projectile 
```
import numpy as np
from scipy.integrate import odeint

# Define the function for the differential equations
def projectile_motion(state, t, d):
    x, vx, z, vz = state
    dxdt = vx
    dvxdt = -d * vx / m
    dzdt = vz
    dvzdt = -g - d * vz / m
    return [dxdt, dvxdt, dzdt, dvzdt]

# Parameters
m = 0.156  # mass of a standard cricket ball in kg
g = 9.81  # acceleration due to gravity in m/s^2
theta = np.deg2rad(45)  # initial angle in radians
d = 0.01  # drag coefficient
x_threshold = 80  # desired displacement in x-direction

# Function to calculate displacement in x-direction
def calculate_displacement(v0):
    # Initial conditions
    x0 = 0
    vx0 = v0 * np.cos(theta)
    z0 = 0
    vz0 = v0 * np.sin(theta)

    # Time array
    t = np.linspace(0, 10, 1000)

    # Solve differential equations
    state0 = [x0, vx0, z0, vz0]
    states = odeint(projectile_motion, state0, t, args=(d,))

    # Find the maximum displacement in x-direction
    max_x_displacement = np.max(states[:, 0])
    return max_x_displacement

# Iterate through different launch velocities
launch_velocities = np.linspace(40, 100, 1000)
valid_launch_velocities = []
for v0 in launch_velocities:
    if calculate_displacement(v0) > x_threshold:
        valid_launch_velocities.append(v0)

print("Launch velocities for which displacement in x is greater than 80 m:")
print(valid_launch_velocities)
```
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("extracted_data.csv")

# Extract the 6th column (assuming zero-indexed, so it's index 5)
column_6 = df.iloc[:, 5]

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(column_6, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of 6th Column")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.grid(True)

# Save the figure as a PDF
plt.savefig("histogram.pdf", format='pdf')

# Show the plot
plt.show()

```
# scatter plot with y=x
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the first CSV file into a pandas DataFrame
df1 = pd.read_csv("file1.csv")

# Read the second CSV file into a pandas DataFrame
df2 = pd.read_csv("file2.csv")

# Extract the 6th column (zero-indexed, so it's index 5) from both DataFrames
column_6_file1 = df1.iloc[:, 5]
column_6_file2 = df2.iloc[:, 5]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(column_6_file1, column_6_file2, color='blue', alpha=0.5, edgecolor='black')

# Add a y=x line
min_val = min(min(column_6_file1), min(column_6_file2))
max_val = max(max(column_6_file1), max(column_6_file2))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.title("Scatter Plot of 6th Column from file1 vs 6th Column from file2")
plt.xlabel("6th Column of file1")
plt.ylabel("6th Column of file2")
plt.grid(True)

# Save the figure as a PDF
plt.savefig("scatter_plot.pdf", format='pdf')

# Show the plot
plt.show()
```
# 
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file into a pandas DataFrame
df1 = pd.read_csv("file1.csv")

# Read the second CSV file into a pandas DataFrame
df2 = pd.read_csv("file2.csv")

# Extract the 6th column (zero-indexed, so it's index 5) from both DataFrames
column_6_file1 = df1.iloc[:, 5]
column_6_file2 = df2.iloc[:, 5]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(column_6_file1, column_6_file2, color='blue', alpha=0.5, edgecolor='black')
plt.title("Scatter Plot of 6th Column from file1 vs 6th Column from file2")
plt.xlabel("6th Column of file1")
plt.ylabel("6th Column of file2")
plt.grid(True)

# Save the figure as a PDF
plt.savefig("scatter_plot.pdf", format='pdf')

# Show the plot
plt.show()
```
# histogram and 
# square scatter plot
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the first CSV file into a pandas DataFrame
df1 = pd.read_csv("file1.csv")

# Read the second CSV file into a pandas DataFrame
df2 = pd.read_csv("file2.csv")

# Extract the 6th column (zero-indexed, so it's index 5) from both DataFrames
column_6_file1 = df1.iloc[:, 5]
column_6_file2 = df2.iloc[:, 5]

# Create a scatter plot
plt.figure(figsize=(8, 8))  # Make the figure square
plt.scatter(column_6_file1, column_6_file2, color='blue', alpha=0.5, edgecolor='black')

# Add a y=x line
min_val = min(min(column_6_file1), min(column_6_file2))
max_val = max(max(column_6_file1), max(column_6_file2))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.title("Scatter Plot of 6th Column from file1 vs 6th Column from file2")
plt.xlabel("6th Column of file1")
plt.ylabel("6th Column of file2")
plt.grid(True)

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Save the figure as a PDF
plt.savefig("scatter_plot.pdf", format='pdf')

# Show the plot
plt.show()
```
# to mark points with abc as 9th column
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the first CSV file into a pandas DataFrame
df1 = pd.read_csv("file1.csv")

# Read the second CSV file into a pandas DataFrame
df2 = pd.read_csv("file2.csv")

# Extract the 6th column (zero-indexed, so it's index 5) from both DataFrames
column_6_file1 = df1.iloc[:, 5]
column_6_file2 = df2.iloc[:, 5]

# Extract the last column (assuming the last column index) from the first DataFrame
last_column_file1 = df1.iloc[:, -1]

# Create a boolean mask for points where the last column in file1 contains "abc"
mask_file1_cs = last_column_file1.str.contains("CS")

# Create a scatter plot
plt.figure(figsize=(8, 8))  # Make the figure square

# Plot points where the last column in file1 contains "CS" in red
plt.scatter(column_6_file1[mask_file1_cs], column_6_file2[mask_file1_cs], color='red', alpha=0.5, edgecolor='black', label='abs')

# Plot points where the last column in file1 does not contain "CS" in blue
plt.scatter(column_6_file1[~mask_file1_cs], column_6_file2[~mask_file1_cs], color='blue', alpha=0.5, edgecolor='black', label='Not abs')

# Add a y=x line
min_val = min(min(column_6_file1), min(column_6_file2))
max_val = max(max(column_6_file1), max(column_6_file2))
plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--')

plt.title("Scatter Plot of 6th Column from file1 vs 6th Column from file2")
plt.xlabel("6th Column of file1")
plt.ylabel("6th Column of file2")
plt.grid(True)

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Add a legend
plt.legend()

# Save the figure as a PDF
plt.savefig("scatter_plot.pdf", format='pdf')

# Show the plot
plt.show()
```
# print the row with minimum value in 6th column in the 3 sets of colours
```
import pandas as pd

# Read the CSV file into a pandas DataFrame
df1 = pd.read_csv("file1.csv")

# Extract the necessary columns
first_column = df1.iloc[:, 0]
sixth_column = df1.iloc[:, 5]

# Define the sets for different colors
red_set = {"Mol_00003", "Mol_00021", "Mol_00088"}
blue_set = {"Mol_00000", "Mol_00001", "Mol_00002", "Mol_00007", "Mol_00008", "Mol_00009", "Mol_00013", "Mol_00014", "Mol_00017", "Mol_00045", "Mol_00046", "Mol_00047", "Mol_00051", "Mol_00055", "Mol_00056", "Mol_00083"}

# Filter the DataFrame for each set and find the row with the minimum value in the 6th column
def print_min_row(set_name, set_values):
    filtered_df = df1[df1.iloc[:, 0].isin(set_values)]
    min_row = filtered_df.loc[filtered_df.iloc[:, 5].idxmin()]
    print(f"Row with minimum value in the 6th column for {set_name}:")
    print(min_row)

# Print the row with minimum value in the 6th column for each set
print_min_row("ER", red_set)
print_min_row("ED", blue_set)
print_min_row("M", set(first_column) - red_set - blue_set)
```
# density plot
```
import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file paths
csv_files = ['data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv', 'data_5.csv']
# Read the 6th column (index 5) from each CSV file into a list
columns_data = []

for file in csv_files:
    df = pd.read_csv(file)
    if df.shape[1] > 5:  # Ensure the file has at least 6 columns
        columns_data.append(df.iloc[:, 5])  # Extract the 6th column

# Concatenate all columns into a single Series for plotting
all_data = pd.concat(columns_data, ignore_index=True)

# Plot density plot
plt.figure(figsize=(10, 6))
all_data.plot(kind='density')
plt.title('Density Plot of the 6th Column from 5 CSV Files')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
```
# sort in sets
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file into a pandas DataFrame
df1 = pd.read_csv("file1.csv")

# Read the second CSV file into a pandas DataFrame
df2 = pd.read_csv("file2.csv")

# Extract the 6th column (index 5) from both DataFrames
column_6_file1 = df1.iloc[:, 5]
column_6_file2 = df2.iloc[:, 5]

# Extract the first column to determine the colors
first_column = df1.iloc[:, 0]

# Define the sets for different colors
red_set = {"Mol_00003", "Mol_00021", "Mol_00088"}
blue_set = {"Mol_00000", "Mol_00001", "Mol_00002", "Mol_00007", "Mol_00008", "Mol_00009", "Mol_00013", "Mol_00014", "Mol_00017", "Mol_00045", "Mol_00046", "Mol_00047", "Mol_00051", "Mol_00055", "Mol_00056", "Mol_00083"}

# Determine the colors and labels based on the first column
colors = []
labels = []
for value in first_column:
    if value in red_set:
        colors.append("darkred")
        labels.append("ER")
    elif value in blue_set:
        colors.append("darkblue")
        labels.append("ED")
    else:
        colors.append("darkgreen")
        labels.append("M")

# Create a scatter plot
plt.figure(figsize=(8, 8))  # Make the figure square

# Plot points
scatter = plt.scatter(column_6_file1, column_6_file2, c=colors, alpha=0.7, edgecolor='black')

# Add a y=x line
min_val = min(min(column_6_file1), min(column_6_file2))
max_val = max(max(column_6_file1), max(column_6_file2))
plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--')

# Add title and labels
plt.title("Scatter Plot of 6th Column from file1 vs 6th Column from file2")
plt.xlabel("6th Column of file1")
plt.ylabel("6th Column of file2")
plt.grid(True)

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Add a legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='E1', markerfacecolor='darkred', markersize=10, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='E2', markerfacecolor='darkblue', markersize=10, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='E3', markerfacecolor='darkgreen', markersize=10, markeredgecolor='black')]
plt.legend(handles=legend_elements, title='Legend')

# Save the figure as a PDF
plt.savefig("scatter_plot.pdf", format='pdf')

# Show the plot
plt.show()
```
```
import pandas as pd

# Read the first and second CSV files into pandas DataFrames
df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")

# Initialize lists to hold the extracted data
col_10_file1 = []
col_6_file1 = []
col_6_file2 = []
differences = []
col_9_file1 = []

# Loop over each row
for i in range(len(df1)):
    # Extract the relevant columns
    col_10_file1.append(df1.iloc[i, 9])
    col_6_file1.append(df1.iloc[i, 5])
    col_6_file2.append(df2.iloc[i, 5])
    differences.append(df1.iloc[i, 5] - df2.iloc[i, 5])
    col_9_file1.append(df1.iloc[i, 8])

# Combine the extracted columns into a single DataFrame
combined_df = pd.DataFrame({
    "10th Column (File 1)": col_10_file1,
    "6th Column (File 1)": col_6_file1,
    "6th Column (File 2)": col_6_file2,
    "Difference": differences,
    "9th Column (File 1)": col_9_file1
})

# Print the combined DataFrame
print("Combined DataFrame:")
print(combined_df)
```

# plot columns
```
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
df = pd.read_csv('your_file.csv')

# Get the column names except for the last column
columns_to_plot = df.columns[:-1]

# Skip the header row while plotting
data = pd.read_csv('your_file.csv', skiprows=1)

# Print some information for debugging
print("Columns to plot:", columns_to_plot)
print("Data head:")
print(data.head())

# Plot each column
for column in columns_to_plot:
    plt.plot(data[column], label=column)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Plot of each column excluding the last column')
plt.legend()
plt.show()
```
# plot DNC
```
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
df = pd.read_csv('your_file.csv')

# Get the column names except for the last column
columns_to_plot = df.columns[:-1]

# Skip the header row while plotting
data = pd.read_csv('your_file.csv', skiprows=1)

# Plot each column
for column in columns_to_plot:
    plt.plot(data[column], label=column)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Plot of each column excluding the last column')
plt.legend()
plt.show()
```
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

hartree2kcalmol = 627.509

def plot_multiple_csvs(csv_files):
    # Ensure there are exactly 4 files, one for each method
    if len(csv_files) != 4:
        raise ValueError("Please provide exactly four CSV files.")

    # Labels for the plot
    labels = ['.....']

    # Generate x-axis values from -0.5 to +0.5 in 101 points
    x = np.linspace(-0.5, 0.5, 101)

    # Read each CSV file and plot its data
    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame, skipping the header row if it exists
        df = pd.read_csv(csv_file, header=0)

        # Number of columns in the DataFrame
        num_columns = df.shape[1]

        for col_index in range(num_columns):
            try:
                column = df.iloc[:, col_index].astype(float).to_numpy()
                column = column - np.min(column)
                column = column * hartree2kcalmol
                plt.plot(x, column, label=f'{labels[i]} col{col_index+1}')
            except ValueError:
                print(f"Skipping non-numeric column: {df.columns[col_index]} in file {csv_file}")

    plt.xlabel('Displacement [Å]')
    plt.ylabel('E [kcal/mol]')
    plt.title('Plot of CSV Columns from Multiple Files')
    plt.legend()
    #plt.ylim(0, 10)
    #plt.xlim(15, 25)
    # Save the plot as a PDF
    plt.savefig('scan_all_methods_multiple_csvs.pdf')
    plt.show()

# Usage example
csv_files = ['energies1.csv', 'energie2.csv', 'en3.csv', 'energi4.csv']  # Replace with the paths to your CSV files
plot_multiple_csvs(csv_files)

```
# Round off
```
def main():
    # Example values
    values = [3.1415926535, 2.7182818284, 1.6180339887, 0.5772156649, 1.4142135623, 2.3025850929, 1.7320508075]

    # First five values rounded to four decimal places
    rounded_values_4_format = [format(value, ".4f") for value in values[:5]]
    rounded_values_4_fstring = [f"{value:.4f}" for value in values[:5]]

    # Last two values rounded to two decimal places
    rounded_values_2_format = [format(value, ".2f") for value in values[5:]]
    rounded_values_2_fstring = [f"{value:.2f}" for value in values[5:]]

    # Combine the results for easier comparison
    rounded_values_format = rounded_values_4_format + rounded_values_2_format
    rounded_values_fstring = rounded_values_4_fstring + rounded_values_2_fstring

    print("Rounded values using format function and f-strings side by side:")
    for i, (value_format, value_fstring) in enumerate(zip(rounded_values_format, rounded_values_fstring), start=1):
        print(f"Value {i}: {value_format} (format) | {value_fstring} (f-string)")

if __name__ == "__main__":
    main()
```
# Read csv file, with certain ponts in x axis
# print in one table, using do loop, 10th column of 1st csv, the 6th column of 1st csv file, 6th column of second csv , difference between the values of the 6th columns of the 2 csv,9th columns of first csv 
```
import pandas as pd

# Read the first and second CSV files into pandas DataFrames
df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")

# Initialize lists to hold the extracted data
col_10_file1 = []
col_6_file1 = []
col_6_file2 = []
differences = []
col_9_file1 = []

# Loop over each row
for i in range(len(df1)):
    # Extract the relevant columns
    col_10_file1.append(df1.iloc[i, 9])
    col_6_file1.append(df1.iloc[i, 5])
    col_6_file2.append(df2.iloc[i, 5])
    differences.append(df1.iloc[i, 5] - df2.iloc[i, 5])
    col_9_file1.append(df1.iloc[i, 8])

# Combine the extracted columns into a single DataFrame
combined_df = pd.DataFrame({
    "10th Column (File 1)": col_10_file1,
    "6th Column (File 1)": col_6_file1,
    "6th Column (File 2)": col_6_file2,
    "Difference": differences,
    "9th Column (File 1)": col_9_file1
})

# Print the combined DataFrame
print("Combined DataFrame:")
print(combined_df)
```
# python program to plot the difference between the first and 7th column of a csv file. Use pandas
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('your_file.csv')

# Assuming your CSV has headers, you can access columns by their names
first_column = df.iloc[:, 0]  # Assuming first column is indexed at 0
seventh_column = df.iloc[:, 6]  # Assuming seventh column is indexed at 6

# Calculate the difference
difference = seventh_column - first_column

# Plot the difference
plt.plot(difference)
plt.title('Difference between First and Seventh Column')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.show()
```
# write a python program which will first go to the template folder where there is opt.com , then in opt.com it will change values of var1 and var2 . var1 will be from 1.3 to 1.5 with 0.01 increment each time. Same with var 2. So for every var1 there would var 21 var2. Now create all the combinations of these in separate input file in separate folders where the name of folders will be Mol_var1_var2 putting the values of v ar1 and var2 
```
import os

# Define the template folder and file
template_folder = 'C:\\Users\\YourName\\Documents\\template'
template_file = 'opt.com'
output_base_folder = 'C:\\Users\\YourName\\Documents\\output'

# Define the ranges and increments for var1 and var2
var1_start = 1.3
var1_end = 1.5
var1_increment = 0.01
var2_start = 1.3
var2_end = 1.5
var2_increment = 0.01

# Read the template file
with open(os.path.join(template_folder, template_file), 'r') as file:
    template_content = file.read()

# Create combinations of var1 and var2
var1_values = [round(var1_start + i * var1_increment, 2) for i in range(int((var1_end - var1_start) / var1_increment) + 1)]
var1_values.append(var1_end)  # Ensure the end value is included

var2_values = [round(var2_start + i * var2_increment, 2) for i in range(int((var2_end - var2_start) / var2_increment) + 1)]
var2_values.append(var2_end)  # Ensure the end value is included

# Remove duplicates and sort the values
var1_values = sorted(set(var1_values))
var2_values = sorted(set(var2_values))

# Generate directories and files for each combination
for var1 in var1_values:
    for var2 in var2_values:
        # Create the directory name
        folder_name = f'Mol_{var1:.2f}_{var2:.2f}'
        folder_path = os.path.join(output_base_folder, folder_name)
        
        # Create the directory if it does not exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Replace variables with actual values
        modified_content = template_content.replace('var1', str(var1)).replace('var2', str(var2))
        
        # Write the modified content to a new opt.com file in the new directory
        new_file_path = os.path.join(folder_path, template_file)
        with open(new_file_path, 'w') as new_file:
            new_file.write(modified_content)

print("All combinations have been created.")
```
# remove duplicates too
# Contour plot
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data.csv')


# Extract x, y, and z values
x = data.iloc[:, 0]  # Assuming x is in the first column
y = data.iloc[:, 1]  # Assuming y is in the second column
z = data.iloc[:, 2]  # Assuming z is in the third column


levels = np.linspace(np.min(z),np.max(z),200)

xi = np.arange(1.3,1.5+0.01,0.01)
yi = np.arange(1.3,1.5+0.01,0.01)

xi,yi = np.meshgrid(xi,yi)
zi = griddata((x,y),z,(xi,yi),method='cubic')
# Aggregate z values for unique combinations of x and y
# z_aggregated = data.groupby([x, y])[z.name].mean().unstack().values

# Create meshgrid for contour plot
# X, Y = np.meshgrid(np.unique(x), np.unique(y))

# Plot contour
contour = plt.contour(xi,yi,zi, levels=levels)
#plt.clabel(contour,inline=True,fontsize=8)

zi = griddata((y,x),z,(xi,yi),method='cubic')
# Aggregate z values for unique combinations of x and y
# z_aggregated = data.groupby([x, y])[z.name].mean().unstack().values

# Create meshgrid for contour plot
# X, Y = np.meshgrid(np.unique(x), np.unique(y))

# Plot contour
contour = plt.contour(xi,yi,zi, levels=levels)

#plt.clabel(contour,inline=True,fontsize=8)


plt.xlabel(data.columns[0])  # Label x-axis with the first column name
plt.ylabel(data.columns[1])  # Label y-axis with the second column name
plt.title('Contour Plot')
plt.show()
```
# extract the 1st column of a.csv and place it as the 3rd column of b.csv and the total merged csv should be a c.csv
```
import csv

# Read from a.csv and extract the 1st column
with open('a.csv', 'r') as file_a:
    reader_a = csv.reader(file_a)
    column_a = [row[0] for row in reader_a]

# Read from b.csv and append the extracted column from a.csv as the 3rd column
with open('b.csv', 'r') as file_b:
    reader_b = csv.reader(file_b)
    rows_b = list(reader_b)

    # Add the extracted column from a.csv as the 3rd column
    for i, row in enumerate(rows_b):
        if i < len(column_a):
            row.insert(2, column_a[i])
        else:
            # If there are more rows in b.csv than in a.csv, add empty values
            row.insert(2, "")

# Write the combined data to c.csv
with open('c.csv', 'w', newline='') as file_c:
    writer = csv.writer(file_c)
    writer.writerows(rows_b)
```
# contour 2
```
# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_wB97XD.csv')

# Extract x, y, and z values
x = data.iloc[:, 0]  
y = data.iloc[:, 1]  
z = data.iloc[:, 2]  


n = np.size(x)

for i in range(n):
    if x[i] != y[i]: 
        x = np.append(x, y[i])
        y = np.append(y, x[i])
        z = np.append(z, z[i])


cmap = plt.cm.BuGn # pellet

dE = 0.001  # contour difference 
numb = int((np.max(z)- np.min(z))/dE) # grid size
levels = np.linspace(np.min(z),np.max(z),numb)



xi = np.linspace(x.min(), x.max(), 1024)
yi = np.linspace(y.min(), y.max(), 1024)


xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')


fig, ax = plt.subplots(figsize=(10, 8))

cp = plt.contourf(xi, yi, zi, levels=levels, cmap=cmap)
# plt.colorbar(cp) # Makrking countour line values


contour = plt.contour(xi, yi, zi,  levels=levels, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8) # Makrking countour line values


plt.title('XXXXXXXXXXXXXXXXXXXXXXXXXX')
plt.xlabel("XXXXXXXXXXXXXXXXXXXXXXXXXX ")
plt.ylabel("XXXXXXXXXXXXXXXXXXXXXXXXXX ")

plt.show()
```
# choose colour palette 
```
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Assuming z is your data
z = np.random.rand(100, 100)  # Example data

cmap = plt.cm.viridis

# Create a new colormap with alpha
alpha = 0.5  # Adjust the alpha value (0: fully transparent, 1: fully opaque)
cmap_colors = cmap(np.arange(cmap.N))
cmap_colors[:, -1] = alpha  # Change the alpha values
cmap_alpha = ListedColormap(cmap_colors)

dE = 0.001  # contour difference 
numb = int((np.max(z) - np.min(z)) / dE)  # grid size
levels = np.linspace(np.min(z), np.max(z), 200)

# Create the contour plot with the custom colormap
plt.contourf(z, levels=levels, cmap=cmap_alpha)
plt.colorbar()
plt.show()
```
```
def main():
    # Example values
    values = [3.1415926535, 2.7182818284, 1.6180339887, 0.5772156649, 1.4142135623, 2.3025850929, 1.7320508075]

    # First five values rounded to four decimal places
    rounded_values_4 = [format(value, ".4f") for value in values[:5]]

    # Last two values rounded to two decimal places
    rounded_values_2 = [format(value, ".2f") for value in values[5:]]

    # Combine the results
    rounded_values = rounded_values_4 + rounded_values_2

    # Print the values side by side with space in between
    print(" ".join(rounded_values))

if __name__ == "__main__":
    main()
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('your_csv_file.csv')

# Extract x, y, and z values
x = data.iloc[:, 0]  # Assuming x is in the first column
y = data.iloc[:, 1]  # Assuming y is in the second column
z = data.iloc[:, 2]  # Assuming z is in the third column

# Aggregate z values for unique combinations of x and y
z_aggregated = data.groupby([x, y])[z.name].mean().unstack().values

# Create meshgrid for contour plot
X, Y = np.meshgrid(np.unique(x), np.unique(y))

# Plot contour
plt.contour(X, Y, z_aggregated)
plt.xlabel(data.columns[0])  # Label x-axis with the first column name
plt.ylabel(data.columns[1])  # Label y-axis with the second column name
plt.title('Contour Plot')

# Save the figure as a PNG file
plt.savefig('contour_plot.png')

# Show the plot
plt.show()
```
```
import os
import re

# Define the directory where the folders are located
root_dir = '/path/to/root/directory'

# Define the regex pattern to extract the numeric value after "="
pattern = re.compile(r'MP2/DEF2-SVP//MP2/DEF2-SVP energy=\s*(-?\d+\.\d+)')

# Function to extract numeric value from opt.out file
def extract_numeric_value(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                numeric_value = float(match.group(1))
                return numeric_value
    return None

# Iterate through folders starting with "Mol"
for folder_name in os.listdir(root_dir):
    if folder_name.startswith("Mol") and os.path.isdir(os.path.join(root_dir, folder_name)):
        folder_path = os.path.join(root_dir, folder_name)
        opt_out_path = os.path.join(folder_path, "opt.out")
        if os.path.exists(opt_out_path):
            numeric_value = extract_numeric_value(opt_out_path)
            if numeric_value is not None:
                print(f"Folder: {folder_name}, Numeric value: {numeric_value}")
            else:
                print(f"No numeric value found in opt.out file of folder {folder_name}")
        else:
            print(f"opt.out file not found in folder {folder_name}")
```
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

hartree2kcalmol = 627.509

def plot_single_csv(csv_file):
    # Read the CSV file into a DataFrame, skipping the header row if it exists
    df = pd.read_csv(csv_file, header=0)

    # Number of columns in the DataFrame
    num_columns = df.shape[1]
    x = np.linspace(-0.2, 0.2, 41)  # Assuming 41 points

    # Plot each column
    for col_index in range(num_columns):
        try:
            column = df.iloc[:, col_index].astype(float).to_numpy()
            column = column - np.min(column)  # Adjust values relative to minimum
            column = column * hartree2kcalmol  # Convert to kcal/mol
            plt.plot(x, column, label=f'Column {col_index+1}')
        except ValueError:
            print(f"Skipping non-numeric column: {df.columns[col_index]} in file {csv_file}")

    plt.xlabel('Index, mode 6')
    plt.ylabel('E [kcal/mol]')
    plt.title('Scan Plot')
    plt.legend()
    plt.grid(True)
    # Save the plot as a PDF
    plt.savefig('scan.pdf')
    plt.show()

# Usage example
csv_file = 'Mol.csv'  # Replace with the path to your CSV file
plot_single_csv(csv_file)
```
```
import csv

# Read from a.csv and extract the 1st column
with open('a.csv', 'r') as file_a:
    reader_a = csv.reader(file_a)
    column_a = [row[0] for row in reader_a]

# Read from b.csv and append the extracted column from a.csv as the 3rd column
with open('b.csv', 'r') as file_b:
    reader_b = csv.reader(file_b)
    rows_b = list(reader_b)

    # Add the extracted column from a.csv as the 3rd column
    for i, row in enumerate(rows_b):
        if i < len(column_a):
            row.insert(2, column_a[i])
        else:
            # If there are more rows in b.csv than in a.csv, add empty values
            row.insert(2, "")

# Write the combined data to c.csv
with open('c.csv', 'w', newline='') as file_c:
    writer = csv.writer(file_c)
    writer.writerows(rows_b)
```
