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
# 
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

hartree2kcalmol = 627.509

def plot_single_csv(csv_file):
    # Label for the plot
    label = 'Single CSV Data'

    # Generate x-axis values from -0.5 to +0.5 in 101 points
    x = np.linspace(-0.5, 0.5, 101)

    # Read the CSV file into a DataFrame, skipping the header row if it exists
    df = pd.read_csv(csv_file, header=0)

    # Number of columns in the DataFrame
    num_columns = df.shape[1]

    for col_index in range(num_columns):
        try:
            column = df.iloc[:, col_index].astype(float).to_numpy()
            column = column - np.min(column)
            column = column * hartree2kcalmol
            plt.plot(x, column, label=f'{label} col{col_index+1}')
        except ValueError:
            print(f"Skipping non-numeric column: {df.columns[col_index]} in file {csv_file}")

    plt.xlabel('Displacement [Å]')
    plt.ylabel('E [kcal/mol]')
    plt.title('Plot of Columns from Single CSV File')
    plt.legend()
    # Save the plot as a PDF
    plt.savefig('single_csv_plot.pdf')
    plt.show()

# Usage example
csv_file = 'energies_wB97XD3_101pts.csv'  # Replace with the path to your CSV file
plot_single_csv(csv_file)
```
# add a column in the left of the already existing column. The column should include values from -2.5 to 2.5 with increment of 0.05 . Like. -2.5,-2.45,-2.45........2.4,2.45 , 2.5. There should be 101 values in that column.
```
import pandas as pd
import numpy as np

# Define the function to add a new column and save to a new CSV
def add_new_column_to_csv(input_csv, output_csv):
    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Create the new column with values from -2.5 to 2.5 in increments of 0.05 and round to 2 decimal places
    new_column = np.round(np.arange(-2.5, 2.55, 0.05), 2)
    
    # Ensure the new column length matches the DataFrame length
    if len(new_column) > len(df):
        new_column = new_column[:len(df)]
    elif len(new_column) < len(df):
        raise ValueError("The new column has fewer values than the number of rows in the DataFrame.")
    
    # Add the new column to the DataFrame at the first position
    df.insert(0, 'New_Column', new_column)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Example usage:
# add_new_column_to_csv('input.csv', 'output.csv')
add_new_column_to_csv('Mol.csv', 'output.csv')
```
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'scan_all_data.csv'
try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Conversion factor from Hartree to eV
conversion_factor = 27.2114

# Define the specific values for the first column
x_values = [-2.5, -2, -1.8, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 1.8, 2, 2.5]

# Filter rows where the first column has the specified values and the 3rd, 4th, and 5th columns have non-missing values
filtered_df = df[df[df.columns[0]].isin(x_values)].dropna(subset=[df.columns[2], df.columns[3], df.columns[4]])

# Extract the relevant columns
x = filtered_df.iloc[:, 0]
y1 = filtered_df.iloc[:, 2] * conversion_factor
y2 = filtered_df.iloc[:, 3] * conversion_factor
y3 = filtered_df.iloc[:, 4] * conversion_factor
y4 = filtered_df.iloc[:, 1] * conversion_factor  # Include the second column and convert it to eV

# Plotting
plt.figure(figsize=(10, 6))

# Plot each y column
plt.plot(x, y1, label='S$_1$', marker='o')
plt.plot(x, y2, label='T$_1$', marker='o')
plt.plot(x, y3, label='STG', marker='o')
plt.plot(x, y4, label='Second Column', marker='o')  # Plot the second column

# Setting the x-axis limits
plt.xlim(-2.5, 2.5)
# plt.ylim(-1, 2.0)

# Adding labels and title
plt.xlabel('DNC')
plt.ylabel('Energy (eV)')
plt.title('Scan plot of the excited states and STG')
plt.legend()

# Show plot
plt.show()
```
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

hartree2ev = 27.2114

def plot_second_column_eV(csv_file):
    # Read the CSV file into a DataFrame, skipping the header row if it exists
    df = pd.read_csv(csv_file, header=0)

    # Extract the second column
    column = df.iloc[:, 1].astype(float).to_numpy()
    column = column - np.min(column)
    column = column * hartree2ev

    # Generate x-axis values from -0.5 to +0.5 in 101 points
    x = np.linspace(-2.5, 2.5, len(column))

    # Plot the data
    plt.plot(x, column)
    plt.xlabel('Displacement [Å]')
    plt.ylabel('E [eV]')
    plt.title('Plot of Second Column from CSV File (in eV)')
    plt.show()

# Usage example
csv_file = 'Mol89_energies_wB97XD3_0.05.csv'  # Replace with the path to your CSV file
plot_second_column_eV(csv_file)
```
# Plot first column
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

hartree2kcalmol = 627.509

def plot_second_column(csv_file):
    # Read the CSV file into a DataFrame, skipping the header row if it exists
    df = pd.read_csv(csv_file, header=0)

    # Extract the second column
    column = df.iloc[:, 1].astype(float).to_numpy()
    column = column - np.min(column)
    column = column * hartree2kcalmol

    # Generate x-axis values from -0.5 to +0.5 in 101 points
    x = np.linspace(-2.5, 2.5, len(column))

    # Plot the data
    plt.plot(x, column)
    plt.xlabel('Displacement [Å]')
    plt.ylabel('E [kcal/mol]')
    plt.title('Plot of Second Column from CSV File')
    plt.show()

# Usage example
csv_file = 'scan_all_data.csv'  # Replace with the path to your CSV file
plot_second_column(csv_file)
```
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file, skipping the first row (header) and any bad lines
file_path = 'scan_all_data.csv'
try:
    df = pd.read_csv(file_path, skiprows=1, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Filter rows where the first column has the value of 0 and the 2nd, 3rd, and 4th columns have non-missing values
filtered_df = df[(df[df.columns[0]] == 0) & (df[df.columns[1]].notna()) & (df[df.columns[2]].notna()) & (df[df.columns[3]].notna())]

# Extract the relevant columns
x = filtered_df.iloc[:, 0]
y1 = filtered_df.iloc[:, 1]
y2 = filtered_df.iloc[:, 2]
y3 = filtered_df.iloc[:, 3]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each y column without markers
plt.plot(x, y1, label='Column 2')
plt.plot(x, y2, label='Column 3')
plt.plot(x, y3, label='Column 4')

# Adding labels and title
plt.xlabel('X-axis (Column 1)')
plt.ylabel('Y-axis (Columns 2, 3, 4)')
plt.title('Plot of Columns 2, 3, and 4 vs Column 1')
plt.legend()

# Show plot
plt.show()
```
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("your_csv_file.csv")

# Filter the DataFrame based on the values in the first column (DNC)
filtered_df = df[df['DNC'].isin([-2.5, -2, -1.8, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 1.8, 2, 2.5])]

# Include the row at DNC=0.0
filtered_df = filtered_df.append(df[df['DNC'] == 0.0])

# Plot the 3rd, 4th, and 5th columns
plt.plot(filtered_df['DNC'], filtered_df['column_3'], label='Column 3')
plt.plot(filtered_df['DNC'], filtered_df['column_4'], label='Column 4')
plt.plot(filtered_df['DNC'], filtered_df['column_5'], label='Column 5')

# Add labels and legend
plt.xlabel('DNC')
plt.ylabel('Values')
plt.title('Plot of Columns 3, 4, and 5')
plt.legend()

# Show the plot
plt.show()
```
#
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

hartree2kcm=627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour.csv')

# Extract x, y, and z values
x = data.iloc[:, 0]  
y = data.iloc[:, 1]  
z = data.iloc[:, 2]  

z=z-np.min(z)
z=z*hartree2kcm


n = np.size(x)

for i in range(n):
    if x[i] != y[i]: 
        x = np.append(x, y[i])
        y = np.append(y, x[i])
        z = np.append(z, z[i])


cmap = plt.cm.viridis # pellet

#dE = 0.01  # contour difference 
#numb = int((np.max(z)- np.min(z))/dE) # grid size

#levels = np.linspace(np.min(z),np.max(z),50)

levels = np.linspace(-1,30,30)



alpha = 0.4  # Adjust the alpha value (0: fully transparent, 1: fully opaque)
cmap_colors = cmap(np.arange(cmap.N))
cmap_colors[:, -1] = alpha  # Change the alpha values
cmap_alpha = ListedColormap(cmap_colors)



xi = np.linspace(x.min(), x.max(), 1024)
yi = np.linspace(y.min(), y.max(), 1024)


xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')


fig, ax = plt.subplots(figsize=(10, 8))

cp = plt.contourf(xi, yi, zi, levels=levels, cmap='terrain')
plt.colorbar(cp) # Makrking countour line values


contour = plt.contour(xi, yi, zi,  levels=levels, colors='black', linewidths=0.5)
# plt.clabel(contour, inline=True, fontsize=8) # Makrking countour line values


plt.title('Contour Plot for Mol 22')
plt.xlabel("$r_1$ [$\AA$]")
plt.ylabel("$r_2$ [$\AA$]")
plt.savefig('contour_.png')
plt.show()
```
```
#!/bin/bash

# Define the root directory where the folders are located
root_dir="/path/to/root/directory"

# Function to extract numeric value from opt.out file
extract_numeric_value() {
    file="$1"
    numeric_value=$(grep -oP 'MP2/DEF2-SVP//MP2/DEF2-SVP energy=\s*\K-?\d+\.\d+' "$file")
    if [ -n "$numeric_value" ]; then
        echo "$numeric_value"
    fi
}

# Iterate through folders starting with "Mol"
for folder in "$root_dir"/Mol*/; do
    if [ -d "$folder" ]; then
        opt_out_file="$folder/opt.out"
        if [ -f "$opt_out_file" ]; then
            numeric_value=$(extract_numeric_value "$opt_out_file")
            echo "$numeric_value"
        fi
    fi
done
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read the CSV file and plot the first column
def plot_first_column(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the first column
    first_column = df.iloc[:, 0]
    
    # Generate x-axis values from -0.5 to 0.5 with 101 points
    x = np.linspace(-0.5, 0.5, 101)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, first_column, label='First Column')
    plt.xlabel('X-axis')
    plt.ylabel('First Column Values')
    plt.title('Scan Plot with Double Well Structure')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
csv_file_path = 'path_to_your_file.csv'
plot_first_column(csv_file_path)
```
```
for f in Mol*; do
  adc2_tzvp_S1S0=$( grep -A10 'Excited state' "$f/all.out" | grep -A10 singlet | grep 'Excitation energy' | awk '{printf "%.3f\n", $3}' | head -1 )
  adc2_tzvp_fosc=$( grep -A10 'Excited state' "$f/all.out" | grep -A10 singlet | grep 'Osc. strength' | awk '{printf "%.3f\n", $3}' | head -1 )
  adc2_tzvp_T1S0=$( grep -A10 'Excited state' "$f/all.out" | grep -A10 triplet | grep 'Excitation energy' | awk '{printf "%.3f\n", $3}' | head -1 )
  adc2_tzvp_T2S0=$( grep -A10 'Excited state' "$f/all.out" | grep -A10 triplet | grep 'Excitation energy' | awk '{printf "%.3f\n", $3}' | head -2 | tail -1 )

  echo "Folder: $f"
  echo "S1S0: $adc2_tzvp_S1S0"
  echo "Oscillator Strength: $adc2_tzvp_fosc"
  echo "T1S0: $adc2_tzvp_T1S0"
  echo "T2S0: $adc2_tzvp_T2S0"
  echo "-----------------------------------"
done
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

hartree2kcm=627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_b3lyp.csv')

# Extract x, y, and z values
x = data.iloc[:, 0]  
y = data.iloc[:, 1]  
z = data.iloc[:, 2]  

z=z-np.min(z)
z=z*hartree2kcm


n = np.size(x)

for i in range(n):
    if x[i] != y[i]: 
        x = np.append(x, y[i])
        y = np.append(y, x[i])
        z = np.append(z, z[i])


cmap = plt.cm.viridis # pellet

#dE = 0.01  # contour difference 
#numb = int((np.max(z)- np.min(z))/dE) # grid size

#levels = np.linspace(np.min(z),np.max(z),50)

levels = np.linspace(-1,30,30)



alpha = 0.4  # Adjust the alpha value (0: fully transparent, 1: fully opaque)
cmap_colors = cmap(np.arange(cmap.N))
cmap_colors[:, -1] = alpha  # Change the alpha values
cmap_alpha = ListedColormap(cmap_colors)



xi = np.linspace(x.min(), x.max(), 1024)
yi = np.linspace(y.min(), y.max(), 1024)


xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')


fig, ax = plt.subplots(figsize=(10, 8))

cp = plt.contourf(xi, yi, zi, levels=levels, cmap='terrain')
plt.colorbar(cp) # Makrking countour line values


contour = plt.contour(xi, yi, zi,  levels=levels, colors='black', linewidths=0.5)
# plt.clabel(contour, inline=True, fontsize=8) # Makrking countour line values
```
```
for f in $( cat dirlist.txt ); do

  # ADC2 results
  adc2_svp_S1S0=$( grep -A10 'Excited state' $f/all.out | grep -A10 singlet | grep 'Excitation energy' | awk '{print $3}' | head -1 )
  adc2_svp_fosc=$( grep -A10 'Excited state' $f/all.out | grep -A10 singlet | grep 'Osc. strength' | awk '{print $3}' | head -1 )
  adc2_svp_T1S0=$( grep -A10 'Excited state' $f/all.out | grep -A10 triplet | grep 'Excitation energy' | awk '{print $3}' | head -1 )
  adc2_svp_T2S0=$( grep -A10 'Excited state' $f/all.out | grep -A10 triplet | grep 'Excitation energy' | awk '{print $3}' | head -2 | tail -1 )
  adc2_svp_S1T1=$( echo $adc2_svp_S1S0 $adc2_svp_T1S0 | awk '{ print $1-$2 }' )
  adc2_svp_T21S1=$( echo $adc2_svp_T1S0 $adc2_svp_S1S0 | awk '{ print 2*$1-$2 }' )
  adc2_svp_T21T2=$( echo $adc2_svp_T1S0 $adc2_svp_T2S0 | awk '{ print 2*$1-$2 }' )
  echo $f,$adc2_svp_S1S0,$adc2_svp_fosc,$adc2_svp_T1S0,$adc2_svp_T2S0,$adc2_svp_S1T1,$adc2_svp_T21S1,$adc2_svp_T21T2 >> adc2_svp.csv

 # adc2_tzvp_S1S0=$( grep -A10 'Excited state' $f/all.out | grep -A10 singlet | grep 'Excitation energy' | awk '{print $3}' | head -1 )
 # adc2_tzvp_fosc=$( grep -A10 'Excited state' $f/all.out | grep -A10 singlet | grep 'Osc. strength' | awk '{print $3}' | head -1 )
 # adc2_tzvp_T1S0=$( grep -A10 'Excited state' $f/all.out | grep -A10 triplet | grep 'Excitation energy' | awk '{print $3}' | head -1 )
 # adc2_tzvp_T2S0=$( grep -A10 'Excited state' $f/all.out | grep -A10 triplet | grep 'Excitation energy' | awk '{print $3}' | head -2 | tail -1 )
 # adc2_tzvp_S1T1=$( echo $adc2_tzvp_S1S0 $adc2_tzvp_T1S0 | awk '{ print $1-$2 }' )
 # adc2_tzvp_T21S1=$( echo $adc2_tzvp_T1S0 $adc2_tzvp_S1S0 | awk '{ print 2*$1-$2 }' )
 # adc2_tzvp_T21T2=$( echo $adc2_tzvp_T1S0 $adc2_tzvp_T2S0 | awk '{ print 2*$1-$2 }' )
 # echo $f,$adc2_tzvp_S1S0,$adc2_tzvp_fosc,$adc2_tzvp_T1S0,$adc2_tzvp_T2S0,$adc2_tzvp_S1T1,$adc2_tzvp_T21S1,$adc2_tzvp_T21T2 >> adc2_tzvp.csv

done

```
```
file=$1


Nat=$(grep 'NAtoms= ' $file | awk '{print $2}' | head -1)

echo $Nat
echo $file
grep -A$(( $Nat+4 )) '        Standard orientation: ' $file | tail -$(( $Nat )) | column -t | awk ' {print " " $2 "  " $4 " " $5 " " $6 }' | sed -e "s/ 1 /H/g" | sed -e "s/ 6 /C/g" | sed -e "s/ 7 /N/g"

```
# create input for adc2
```
dirs=$(cat dirlist.txt)

for dir in $dirs; do

  mkdir $dir

  cat template/all1.com  > all.com
  Nat=$( grep -A2 'Current geometry (xyz format, in Angstrom)' ../Contourplot_MP2/$dir/opt.out |  tail -1 | awk '{print $1}' )
  grep -A$(( $Nat+3 )) 'Current geometry (xyz format, in Angstrom)' ../Contourplot_MP2/$dir/opt.out | tail -$Nat >> all.com
  cat template/all2.com >> all.com

  mv all.com $dir

done
```
# scan normal mode 
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'scan_all_data.csv'
try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Conversion factor from Hartree to eV
#conversion_factor = 27.2114

# Define the specific values for the first column
x_values = [-5,-2.5, -2, -1.8, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 1.8, 2, 2.5,5]

# Filter rows where the first column has the specified values and the 3rd, 4th, and 5th columns have non-missing values
filtered_df = df[df[df.columns[0]].isin(x_values)].dropna(subset=[df.columns[2], df.columns[3], df.columns[4]])

# Extract the relevant columns
x = filtered_df.iloc[:, 0]
y1 = filtered_df.iloc[:, 2]
y2 = filtered_df.iloc[:, 3]
y3 = filtered_df.iloc[:, 4]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each y column
plt.plot(x, y1 , label='S$_1$', marker='o')
plt.plot(x, y2 , label='T$_1$', marker='o')
plt.plot(x, y3 , label='STG', marker='o')

# Setting the x-axis limits
plt.xlim(-5, 5)
# plt.ylim(-1, 2.0)

# Adding labels and title
plt.xlabel('aaaa')
plt.ylabel('Energy (eV)')
plt.title('Scan plot of the excited states and STG')
plt.legend()
plt.savefig('ScanP')
# Show plot
plt.show()
```
# Merge 2 csv into a new csv
```
import csv

# Read from a.csv and extract the 1st column
with open('scf.csv', 'r') as file_a:
    reader_a = csv.reader(file_a)
    column_a = [row[0] for row in reader_a]

# Read from b.csv and append the extracted column from a.csv as the 3rd column
with open('var_values.csv', 'r') as file_b:
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
with open('contour.csv', 'w', newline='') as file_c:
    writer = csv.writer(file_c)
    writer.writerows(rows_b)
```
# print numbers with fixed interval
```
# Define the start, stop, and interval
start = -5
stop = 5
interval = 0.05

# Use a loop to generate the numbers and print each one
current = start
while current <= stop:
    print(f"{current:.2f}")
    current += interval
```
# extract del E
```
au2kjm=2625.499618335386


xx=$(grep 'CCSD/cc-pVTZ//CCSD/cc-pVTZ energy=' a/opt.out | awk '{print $3}' )
yy=$(grep 'CCSD/cc-pVTZ//CCSD/cc-pVTZ energy=' abc/opt.out | awk '{print $3}' )

echo "CCSD/VTZ"
echo $xx
echo $yy
awk -v xxx="$xx" -v xxx="$xx" -v conv="$au2kjm" 'BEGIN {printf "%5.1f\n", (xx - yy) * conv}'
```
# double well
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read the CSV file and plot the first column
def plot_first_column(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the first column
    first_column = df.iloc[:, 0]
    
    # Generate x-axis values from -0.5 to 0.5 with 101 points
    x = np.linspace(-0.5, 0.5, 101)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, first_column, label='First Column')
    plt.xlabel('X-axis')
    plt.ylabel('First Column Values')
    plt.title('Scan Plot with Double Well Structure')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
csv_file_path = 'path_to_your_file.csv'
plot_first_column(csv_file_path)
```
```
import os
import csv

# Define the base path where the Mol directories are located
base_path = '/home/atreyee/Project_SOJT_Azaphenalenes/2APC2V_Mol_004/Contourplot_MP2'  # Change to the actual path where Mol directories are located

# Define the output CSV file path
csv_file_path = os.path.join(base_path, 'var_values.csv')

# Prepare a list to store the rows of the CSV
csv_rows = []

# Scan the base directory for folders starting with 'Mol_'
for folder_name in os.listdir(base_path):
    if folder_name.startswith('Mol_'):
        try:
            # Extract var1 and var2 from the folder name
            _, var1, var2 = folder_name.split('_')
            var1 = float(var1)
            var2 = float(var2)

            # Append the values to the csv_rows list
            csv_rows.append([var1, var2])
        except ValueError:
            print(f"Skipping folder with unexpected name format: {folder_name}")

# Sort the rows first by var1 and then by var2
csv_rows.sort()

# Write the rows to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header
    csv_writer.writerow(['var1', 'var2'])
    # Write the rows
    csv_writer.writerows(csv_rows)

print(f"CSV file has been created at {csv_file_path}.")
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

# Read lines from input file
with open('scan_wB97XD3_energy.out', 'r') as file:
    lines = file.readlines()

# List to store the values under the "DNC" column
values = []

# Flags to track when we are within the desired block
in_block = False

# Extract values under "DNC" column from lines within the block
for line in lines:
    line = line.strip()
    
    # Check if the block starts
    if line.startswith('------------------------------------------------------------'):
        in_block = not in_block  # Toggle the in_block flag
        continue
    
    # If we are in the block and the line has the correct format
    if in_block:
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            try:
                # Try to convert the second part to a float to ensure it's a numeric value
                dnc_value = float(parts[1])
                values.append(dnc_value)
            except ValueError:
                continue  # Skip lines that cannot be converted to float

# Ensure we only take the first 41 values (41 trajectory steps)
values = values[:41]

# Write values to a CSV file
with open('values.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Value'])  # Write header
    writer.writerows(map(lambda x: [x], values))

print("Values saved to values.csv")
```
# normal mode scan
```
x = np.linspace(-0.2, 0.2, 41)
# Plot each column
for col_index in range(num_columns):
    try:
        column = df.iloc[:, col_index].astype(float).to_numpy()
        column = column - np.min(column)
        column = column * hartree2kcalmol
        plt.plot(x, column, label='k')
    except ValueError:
        print(f"Skipping non-numeric column: {df.columns[col_index]} in file {csv_file}")

plt.xlabel('Index, mode 6')
plt.ylabel('E [kcal/mol]')
plt.title('Scan Plot')
plt.legend()
#plt.ylim(0, 10)
#plt.xlim(15, 25)
# Save the plot as a PDF
plt.savefig('scan.pdf')
plt.show()
```
# plot horizontal lines
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
# calculate atomization energies and plot histogram
```
import pandas as pd
import matplotlib.pyplot as plt

# Define the constants
x = -0.777
y = -3.88
z = -6.99

# Path to the CSV file
csv_file_path = 'data.csv'

# Read the CSV file without headers
df = pd.read_csv(csv_file_path, header=None)

# Ensure the DataFrame has exactly 4 columns
if df.shape[1] != 4:
    raise ValueError("CSV file must have exactly 4 columns")

# Perform the calculation: (1st_column * x + 2nd_column * y + 3rd_column * z) - 4th_column
df['calculation'] = (df[0] * x + df[1] * y + df[2] * z) - df[3]

# Plot the histogram of the calculated values
plt.figure(figsize=(10, 6))
plt.hist(df['calculation'], bins=30, edgecolor='black')
plt.title('Histogram of Calculated Values')
plt.xlabel('Calculated Value')
plt.ylabel('Frequency')
plt.grid(True)

# Save the figure as a PDF
plt.savefig('histogram.pdf', format='pdf')

# Show the plot
plt.show()
```
# colorcoding histogram
```
import pandas as pd
import matplotlib.pyplot as plt

# Define the constants
x = -0.502112655155
y = -37.844964184283
z = -54.589569458889

# Path to the CSV file
csv_file_path = 'data.csv'

# Read the CSV file without headers
df = pd.read_csv(csv_file_path, header=None)

# Ensure the DataFrame has exactly 4 columns
if df.shape[1] != 4:
    raise ValueError("CSV file must have exactly 4 columns")

# Perform the calculation: (1st_column * x + 2nd_column * y + 3rd_column * z) - 4th_column
df['calculation'] = (df[0] * x + df[1] * y + df[2] * z) - df[3]

# Define row indices for each color
red_rows = [0]  # 0-based index for row 1
green_rows = [3, 21, 88]  # 0-based indices for rows 4, 22, 89
blue_rows = [1, 2, 7, 8, 9, 13, 14, 17, 45, 46, 47, 51, 55, 56, 53]  # 0-based indices for specified rows

# Separate the calculations by color groups
red_values = df['calculation'].iloc[red_rows]
green_values = df['calculation'].iloc[green_rows]
blue_values = df['calculation'].iloc[blue_rows]
orange_values = df['calculation'].drop(red_rows + green_rows + blue_rows)

# Plot the histogram with different colors
plt.figure(figsize=(10, 6))
plt.hist(red_values, bins=30, color='red', edgecolor='black', alpha=0.5, label='Red Rows')
plt.hist(green_values, bins=30, color='darkgreen', edgecolor='black', alpha=0.5, label='Green Rows')
plt.hist(blue_values, bins=30, color='blue', edgecolor='black', alpha=0.5, label='Blue Rows')
plt.hist(orange_values, bins=30, color='orange', edgecolor='black', alpha=0.2, label='Orange Rows')

# Add title and labels
plt.title('Histogram of Calculated Values with Different Colors')
plt.xlabel('Calculated Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save the figure as a PDF
plt.savefig('histogram.pdf', format='pdf')

# Show the plot
plt.show()
```
#
```
for d in Mol_*; do
  f=$d/geom_S0.xyz
  NH=$( grep '^H ' $f  | wc -l )
  NC=$( grep '^C ' $f  | wc -l )
  NN=$( grep '^N ' $f  | wc -l )
  energy=$( bzgrep 'FINAL SINGLE POINT ENERGY' $d/opt.out.bz2| tail -1 | awk '{print $5}' )
  echo $NH,$NC,$NN,$energy
done
```
# extract trajectories
```
echo "s1 s2 s3 s4 s5 s6 t1 t2 t3 t4 t5 t6 S0" > tda.csv

# Loop through all folders starting with "traj_"
for i in $(seq -w 1 41); do
    folder='traj_'$i
    echo "Processing folder: $folder"
    if [ -d "$folder" ]; then
        cd "$folder" || exit
        # Execute the extraction commands in the current folder
        bzgrep 'STATE  1:  E' tda.out.bz2 | grep '2.000000' | awk '{print $6}' > t1.txt
        bzgrep 'STATE  2:  E' tda.out.bz2 | grep '2.000000' | awk '{print $6}' > t2.txt
        bzgrep 'STATE  3:  E' tda.out.bz2 | grep '2.000000' | awk '{print $6}' > t3.txt
        bzgrep 'STATE  4:  E' tda.out.bz2 | grep '2.000000' | awk '{print $6}' > t4.txt
        bzgrep 'STATE  5:  E' tda.out.bz2 | grep '2.000000' | awk '{print $6}' > t5.txt
        bzgrep 'STATE  6:  E' tda.out.bz2 | grep '2.000000' | awk '{print $6}' > t6.txt
        bzgrep 'STATE  1:  E' tda.out.bz2 | grep '0.000000' | awk '{print $6}' > s1.txt
        bzgrep 'STATE  2:  E' tda.out.bz2 | grep '0.000000' | awk '{print $6}' > s2.txt
        bzgrep 'STATE  3:  E' tda.out.bz2 | grep '0.000000' | awk '{print $6}' > s3.txt
        bzgrep 'STATE  4:  E' tda.out.bz2 | grep '0.000000' | awk '{print $6}' > s4.txt
        bzgrep 'STATE  5:  E' tda.out.bz2 | grep '0.000000' | awk '{print $6}' > s5.txt
        bzgrep 'STATE  6:  E' tda.out.bz2 | grep '0.000000' | awk '{print $6}' > s6.txt
        bzgrep 'Total Energy       : ' tda.out.bz2 | awk '{print $6+14952.75064}' > S0.txt
        # Paste the extracted data into a temporary file
        paste -d ' ' s1.txt s2.txt s3.txt s4.txt s5.txt s6.txt t1.txt t2.txt t3.txt t4.txt t5.txt t6.txt S0.txt >> ../tda.csv
        # Clean up temporary files
        rm -f s*.txt t*.txt S0.txt
        cd ..
    fi
done

echo "Extraction completed. Results saved in tda.csv"
```
# print the folders names the values
```
import os
import csv

# Define the base path where the Mol directories are located
base_path = '/home/atreyee/folder'  # Change to the actual path where Mol directories are located

# Define the output CSV file path
csv_file_path = os.path.join(base_path, 'var_values.csv')

# Prepare a list to store the rows of the CSV
csv_rows = []

# Scan the base directory for folders starting with 'Mol_'
for folder_name in os.listdir(base_path):
    if folder_name.startswith('Mol_'):
        try:
            # Extract var1 and var2 from the folder name
            _, var1, var2 = folder_name.split('_')
            var1 = float(var1)
            var2 = float(var2)

            # Append the values to the csv_rows list
            csv_rows.append([var1, var2])
        except ValueError:
            print(f"Skipping folder with unexpected name format: {folder_name}")

# Sort the rows first by var1 and then by var2
csv_rows.sort()

# Write the rows to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header
    csv_writer.writerow(['var1', 'var2'])
    # Write the rows
    csv_writer.writerows(csv_rows)

print(f"CSV file has been created at {csv_file_path}.")
```
#
```
#t directory where the folders are located
root_dir="/home/atreyee/Project_SOJT_Azaphenalenes/cc_basis/1AP_D3h_001/Contourplot_CCSDT_at_MP2"

# Function to extract numeric value from opt.out file
extract_numeric_value() {
    file="$1"
    numeric_value=$(grep -oP 'CCSD(T)/cc-pVDZ//CCSD(T)/cc-pVDZ energy=\s*\K-?\d+\.\d+' "$file")
    if [ -n "$numeric_value" ]; then
        echo "$numeric_value"
    fi
}

# Iterate through folders starting with "Mol"
for folder in "$root_dir"/Mol*/; do
    if [ -d "$folder" ]; then
        opt_out_file="$folder/opt.out"
        if [ -f "$opt_out_file" ]; then
            numeric_value=$(extract_numeric_value "$opt_out_file")
            echo "$numeric_value"
        fi
    fi
done
```
# adc2 input
```
dirs=$(cat dirlist.txt)

for dir in $dirs; do

  mkdir $dir

  cat template/all1.com  > all.com
  Nat=$( grep -A2 'Current geometry (xyz format, in Angstrom)' ../Contourplot_MP2/$dir/opt.out |  tail -1 | awk '{print $1}' )
  grep -A$(( $Nat+3 )) 'Current geometry (xyz format, in Angstrom)' ../Contourplot_MP2/$dir/opt.out | tail -$Nat >> all.com
  cat template/all2.com >> all.com

  mv all.com $dir

done
```
# all combinations of smiles
```


# Words to fill in the brackets
groups = ['','F','C','N','O','C(C)(C)(C)','N(C)(C)','O(C)'] 

Ngrps=len(groups)

file1=open('a_sub.smi','w')

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
            mol='C1'+groupi+'=NC2=NC'+groupj+'=NC3=NC'+groupk+'=NC(=N1)N23'
            name='Mol_'+str(i)+'_'+str(j)+'_'+str(k)
            file1.write(mol+' '+name+'\n')
    

file1.close()
```
# extract energy
```
numeric_value=$(grep -oP 'CCSD(T)/cc-pVDZ energy=\s*-?\d+\.\d+' "$file" | grep -oP '-?\d+\.\d+')
```
# adc2 inp and folder creation with fortran instead of python
```
!gfortran prepinp_adc2.f90 -o prepinp_adc2.x
!./prepinp_adc2.x 
program main

  implicit none

  integer, parameter            :: Nmol = 120
  character(len=500)            :: cmd, filedir
  character(len=100)            :: line, filename, smi, title
  integer                       :: Nat, iat, imol, tmp, iq, im
  character(len=1), allocatable :: sym(:)
  double precision              :: beta, dipole
  double precision, allocatable :: R(:,:)

  open(unit=100, file='geom_DFT_S0_all.xyz')

  do imol = 1, Nmol

    read(100,*) Nat
    read(100,*) title
    allocate(sym(1:Nat), R(1:Nat,1:3))
    do iat = 1, Nat
      read(100,*) sym(iat), R(iat,1:3)
    enddo

    open(unit=101, file='all.com')
    write(101,'(a)')'$molecule'
    write(101,'(2i3)')0, 1
    do iat = 1, Nat
        write(101,'(a,3f15.8)') sym(iat), R(iat,1:3)
    enddo
    write(101,'(a)')'$end'
    write(101,'(a)')''
    write(101,'(a)')'$rem                                 '
    write(101,'(a)')'jobtype             sp               '
    write(101,'(a)')'method              adc(2)           '
    write(101,'(a)')'basis               cc-pVTZ        '
    write(101,'(a)')'aux_basis           rimp2-cc-pVTZ  '
    write(101,'(a)')'mem_total           64000            '
    write(101,'(a)')'mem_static          1000             '
    write(101,'(a)')'maxscf              1000             '
    write(101,'(a)')'cc_symmetry         false            '
    write(101,'(a)')'ee_singlets         3                '
    write(101,'(a)')'ee_triplets         3                '
    write(101,'(a)')'sym_ignore          true             '
    write(101,'(a)')'ADC_DAVIDSON_MAXITER 300'
    write(101,'(a)')'ADC_DAVIDSON_CONV 5'
    write(101,'(a)')'$end                                 '
    close(101)

    deallocate(sym, R)                                                                                                                                                          1,1           Top

 write(cmd,'(a,i5.5)')'mkdir Mol_', imol
    call system(trim(cmd))

    write(cmd,'(a,i5.5)')'mv all.com Mol_', imol
    call system(trim(cmd))

  enddo

  close(100)

end program main

```
# distance from smiles 
```
import re
import bz2
from math import sqrt
from rdkit import Chem

# Function to extract the bz2 file
def extract_bz2_file(bz2_filename, extracted_filename):
    with bz2.BZ2File(bz2_filename, 'rb') as file:
        content = file.read()
    with open(extracted_filename, 'wb') as file:
        file.write(content)

# Function to parse the ORCA output file
def parse_orca_output(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the Cartesian coordinates section
    start = end = 0
    for i, line in enumerate(lines):
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            start = i + 2  # coordinates usually start two lines below this line
            break
    
    for i in range(start, len(lines)):
        if lines[i].strip() == '' or lines[i].startswith('-----'):
            end = i
            break
    
    coordinates = []
    for line in lines[start:end]:
        parts = re.split(r'\s+', line.strip())
        atom = parts[0]
        x, y, z = map(float, parts[1:])
        coordinates.append((atom, x, y, z))
    
    return coordinates

# Function to calculate distance between two atoms
def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1:]
    x2, y2, z2 = atom2[1:]
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# File names
bz2_filename = 'opt.out.bz2'
extracted_filename = 'opt.out'

# Extract the bz2 file
extract_bz2_file(bz2_filename, extracted_filename)

# Parse the extracted ORCA output file
coordinates = parse_orca_output(extracted_filename)

# Interpret SMILES string with RDKit
smiles = 'C1=NC2=NC=NC3=NC=NC(=N1)N23'
mol = Chem.MolFromSmiles(smiles)

# Get the atomic indices in the correct order
atom_indices = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}

# Map indices to coordinates
atom_coords = {idx: coordinates[idx] for idx in atom_indices}

# Define atom pairs for distances
pairs = {
    'r1': (0, 8),  # atom 1 (index 0) to atom 9a (index 8)
    'r2': (2, 9),  # atom 3 (index 2) to atom 3a (index 9)
    'r3': (9, 3),  # atom 3a (index 9) to atom 4 (index 3)
    'r4': (5, 10), # atom 6 (index 5) to atom 6a (index 10)
    'r5': (10, 6), # atom 6a (index 10) to atom 7 (index 6)
    'r6': (8, 7),  # atom 9 (index 8) to atom 9a (index 7)
}

# Calculate distances
distances = {key: calculate_distance(atom_coords[pair[0]], atom_coords[pair[1]]) for key, pair in pairs.items()}

# Print distances
for key, dist in distances.items():
    print(f'{key} distance: {dist:.4f} Å')
```
# 
```
import re
import bz2
from math import sqrt
from rdkit import Chem

# Function to extract the bz2 file
def extract_bz2_file(bz2_filename, extracted_filename):
    with bz2.BZ2File(bz2_filename, 'rb') as file:
        content = file.read()
    with open(extracted_filename, 'wb') as file:
        file.write(content)

# Function to parse the ORCA output file
def parse_orca_output(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the Cartesian coordinates section
    start = end = 0
    for i, line in enumerate(lines):
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            start = i + 2  # coordinates usually start two lines below this line
            break
    
    for i in range(start, len(lines)):
        if lines[i].strip() == '' or lines[i].startswith('-----'):
            end = i
            break
    
    coordinates = []
    for line in lines[start:end]:
        parts = re.split(r'\s+', line.strip())
        atom = parts[0]
        x, y, z = map(float, parts[1:])
        coordinates.append((atom, x, y, z))
    
    return coordinates

# Function to calculate distance between two atoms
def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1[1:]
    x2, y2, z2 = atom2[1:]
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# File names
bz2_filename = 'opt.out.bz2'
extracted_filename = 'opt.out'

# Extract the bz2 file
extract_bz2_file(bz2_filename, extracted_filename)

# Parse the extracted ORCA output file
coordinates = parse_orca_output(extracted_filename)

# Interpret SMILES string with RDKit
smiles = 'C1=NC2=NC=NC3=NC=NC(=N1)N23'
mol = Chem.MolFromSmiles(smiles)

# Get the atomic indices in the correct order
atom_indices = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}

# Map indices to coordinates
atom_coords = {idx: coordinates[idx] for idx in atom_indices}

# Define atom pairs for distances
pairs = {
    'r1': (0, 8),  # atom 1 (index 0) to atom 9a (index 8)
    'r2': (2, 9),  # atom 3 (index 2) to atom 3a (index 9)
    'r3': (9, 3),  # atom 3a (index 9) to atom 4 (index 3)
    'r4': (5, 10), # atom 6 (index 5) to atom 6a (index 10)
    'r5': (10, 6), # atom 6a (index 10) to atom 7 (index 6)
    'r6': (8, 7),  # atom 9 (index 8) to atom 9a (index 7)
}

# Calculate distances
distances = {key: calculate_distance(atom_coords[pair[0]], atom_coords[pair[1]]) for key, pair in pairs.items()}

# Print distances
for key, dist in distances.items():
    print(f'{key} distance: {dist:.4f} Å')
```
```
for f in */*wB97XD*/opt.log; do echo $f; grep -A8 'Diagonal vibrational polarizability' $f | tail -2; echo "===================================="; done
for f in */*B3LYP*/opt.log; do echo $f; grep -A8 'Diagonal vibrational polarizability' $f | tail -2; echo "===================================="; done
```
# heading in jupyter notebook
```
# <span style="color:blue">Heading 1</span>
## <span style="color:green">Heading 2</span>
### <span style="color:red">Heading 3</span>
#### <span style="color:purple">Heading 4</span>
##### <span style="color:orange">Heading 5</span>
###### <span style="color:brown">Heading 6</span>
```
# query for pymoldis
```
import pymoldis
import pandas as pd

df=pymoldis.get_data('bigqm7w_S1T1')

S1T1_DFT=df['S1_SCSPBEQIDH(eV)'] - df['T1_SCSPBEQIDH(eV)']

NEntries=15

SmallGap_DFT_vals=S1T1_DFT.nsmallest(NEntries) 

SMIs=df.iloc[SmallGap_DFT_vals.index]['SMI']

result = pd.concat([SMIs, SmallGap_DFT_vals], axis=1)
result.columns = ['SMI','S1-T1(eV)']
print(result)
```
# query 5
```
diff_dft=df['S1_SCSPBEQIDH(eV)'] - df['T1_SCSPBEQIDH(eV)']
diff_adc2=df['S1_ADC2(eV)'] - df['T1_ADC2(eV)']

N_smallest=5
entries_dft=df.iloc[diff_dft.abs().nsmallest(N_smallest).index]
entries_adc2=df.iloc[diff_adc2.abs().nsmallest(N_smallest).index]
```
# Query 7
```
import pymoldis

df=pymoldis.get_data('bigqm7w_S1T1')

lower_bound=3.0
upper_bound=4.0

filtered_df=df[(df['S1_ADC2(eV)'] >= lower_bound) & (df['S1_ADC2(eV)'] <= upper_bound) & 
                 (df['T1_ADC2(eV)'] >= lower_bound) & (df['T1_ADC2(eV)'] <= upper_bound)]

filtered_df=filtered_df[['SMI','S1_ADC2(eV)','T1_ADC2(eV)','f01_ADC2(au)']]

print(filtered_df.describe())
```
#
```
def svg_from_smiles(SMI):
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from io import StringIO

    mol=Chem.MolFromSmiles(SMI)
    drawer=rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg=drawer.GetDrawingText()

    return svg
```
# extract all energies
```
#t directory where the folders are located
root_dir="/home/atreyee/Project_SOJT_Azaphenalenes/cc_basis/5AP_C2v_056/Contourplot_MP2"

# Function to extract numeric value from opt.out file
extract_numeric_value() {
    file="$1"
    numeric_value=$(grep -oP 'MP2/cc-pVDZ//MP2/cc-pVDZ energy=\s*\K-?\d+\.\d+' "$file")
    if [ -n "$numeric_value" ]; then
        echo "$numeric_value"
    fi
}

# Iterate through folders starting with "Mol"
for folder in "$root_dir"/Mol*/; do
    if [ -d "$folder" ]; then
        opt_out_file="$folder/opt.out"
        if [ -f "$opt_out_file" ]; then
            numeric_value=$(extract_numeric_value "$opt_out_file")
            echo "$numeric_value"
        fi
    fi
done
```
#
```
# Define the sets as dictionaries for easier manipulation
set_a = {
    'R1': 1.42611145, 'R2': 1.40620982, 'A1': 118.64685962, 
    'R3': 1.41181658, 'A2': 118.07001826, 'R4': 2.77910173, 
    'R5': 1.36541130, 'A3': 120.87669348, 'R6': 1.33487581, 
    'A4': 114.93433118, 'R7': 1.34563290, 'A5': 119.91550111, 
    'R8': 1.09360603, 'A6': 116.73651168, 'R9': 3.87659159, 
    'R10': 1.09954983, 'A7': 115.65703614
}

set_b = {
    'R1': 1.4088, 'R2': 1.3905, 'A1': 118.9429, 
    'R3': 1.3883, 'A2': 118.3252, 'R4': 2.7388, 
    'R5': 1.3459, 'A3': 121.6491, 'R6': 1.3114, 
    'A4': 116.2682, 'R7': 1.3242, 'A5': 119.5042, 
    'R8': 1.0776, 'A6': 117.0842, 'R9': 3.8204, 
    'R10': 1.0848, 'A7': 116.1637
}

set_c = {
    'R1': 1.4132, 'R2': 1.3961, 'A1': 118.8895, 
    'R3': 1.3964, 'A2': 118.3005, 'R4': 2.7535, 
    'R5': 1.3508, 'A3': 121.4956, 'R6': 1.3188, 
    'A4': 115.9481, 'R7': 1.3308, 'A5': 119.7015, 
    'R8': 1.0875, 'A6': 116.8684, 'R9': 3.8449, 
    'R10': 1.0951, 'A7': 115.9548
}

# Function to perform the operation A + (B - C)
def compute(set_a, set_b, set_c):
    result = {}
    for key in set_a:
        result[key] = set_a[key] + (set_b[key] - set_c[key])
    return result

# Perform the computation
result = compute(set_a, set_b, set_c)

# Print the results
for key, value in result.items():
    print(f"{key}: {value:.8f}")

```
# 
```
import pandas as pd
import matplotlib.pyplot as plt

# Define the energies and custom labels
data = {
    'Label': ['a', '2', '3', '4', '6', '8'],
    'Energy': [
        -385.691580683,
        -770.860688368 / 2,
        -771.336917778 / 2,
        -771.380824354 / 2,
        -771.382283738 / 2,
        -771.382816488 / 2
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the line passing through the first point
plt.figure(figsize=(10, 6))
plt.plot([1, len(df)], [df['Energy'][0], df['Energy'][0]], color='r', label='Line through Point a')

# Plot the curve joining the other points
plt.plot(range(2, len(df) + 1), df['Energy'][1:], marker='o', linestyle='-', color='b', label='Curve through other points')

# Add labels to each point
for i, row in df.iterrows():
    plt.text(i + 1, row['Energy'], f'{row["Label"]} ({row["Energy"]:.6f})', fontsize=10, ha='right')

# Add title and labels
plt.title('Energy Plot')
plt.xlabel('Index')
plt.ylabel('Energy')
plt.legend()

# Save the figure as a PDF
plt.savefig('energies_plot.pdf')

# Show the plot
plt.show()
```
```
# Define the values of sets a, b, c, d, e
a_values = {
    'R1': 1.39431687, 'R21': 1.45490825, 'A11': 115.45743707,
    'R3': 2.80662976, 'A3': 60.55500817,
    'R22': 1.37022431, 'A12': 119.55983802,
    'R41': 1.08535742, 'A21': 116.71005308,
    'R42': 1.08026741, 'A22': 119.96929245
}

b_values = {
    'R1': 1.39389115, 'R21': 1.45482650, 'A11': 115.45373227,
    'R3': 2.80563196, 'A3': 60.55172498,
    'R22': 1.36968614, 'A12': 119.55860443,
    'R41': 1.08513133, 'A21': 116.70819858,
    'R42': 1.07900157, 'A22': 119.96483702
}

c_values = {
    'R1': 1.39486845, 'R21': 1.45491858, 'A11': 115.50106856,
    'R3': 2.80815213, 'A3': 60.56083178,
    'R22': 1.37324760, 'A12': 119.50711226,
    'R41': 1.06624390, 'A21': 116.74865827,
    'R42': 1.08129777, 'A22': 119.95459728
}

d_values = {
    'R1': 1.39982160, 'R21': 1.46288508, 'A11': 115.32151197,
    'R3': 2.81884208, 'A3': 60.59657498,
    'R22': 1.37203124, 'A12': 119.60261270,
    'R41': 1.08908183, 'A21': 116.59657498,
    'R42': 1.08444094, 'A22': 119.07456932
}

e_values = {
    'R1': 1.40012160, 'R21': 1.47448508, 'A11': 115.98121197,
    'R3': 2.81734208, 'A3': 60.23197498,
    'R22': 1.36083124, 'A12': 119.92841270,
    'R41': 1.09008183, 'A21': 116.23197498,
    'R42': 1.08204094, 'A22': 120.55276932
}

# Compute p = (a - b), q = (a - c), r = (a - d), s = (a - e)
p_values = {key: a_values[key] - b_values[key] for key in a_values}
q_values = {key: a_values[key] - c_values[key] for key in a_values}
r_values = {key: a_values[key] - d_values[key] for key in a_values}
s_values = {key: a_values[key] - e_values[key] for key in a_values}

# Print the values in separate columns
print("{:<15} {:<15} {:<15} {:<15}".format("Parameter", "p", "q", "r", "s"))
for key in a_values:
    print("{:<15} {:<15.8f} {:<15.8f} {:<15.8f} {:<15.8f}".format(key, p_values[key], q_values[key], r_values[key], s_values[key]))
```
#
```
import os
import shutil

# List of method names
methods = [
    'B3LYP', 'BHandHLYP', 'CAM-B3LYP', 'M06', 'M062X', 'M06HF', 'M11L',
    'BP86', 'TPSSTPSS', 'WB97XD', 'B97D', 'PBEPBE', 'PBE1PBE', 'HSEH1PBE', 'LC-BLYP', 'LC-wHPBE'
]

# Create folders and copy template file
template_file = 'tddft.com'
for method in methods:
    # Create folder if it doesn't exist
    folder_name = method
    os.makedirs(folder_name, exist_ok=True)

    # Copy template file to folder
    shutil.copy(template_file, folder_name)

    # Replace placeholder in template file
    with open(os.path.join(folder_name, template_file), 'r+') as file:
        content = file.read()
        updated_content = content.replace('xxxx', method)
        file.seek(0)
        file.write(updated_content)
        file.truncate()

print("Folders created and template files copied and updated successfully.")

```
# to extract values from named foldes. It will extract S and T from tddft.log finally
```
import pandas as pd
import matplotlib.pyplot as plt
import string

# Constants
hc_e_conversion_factor = 1.239841984e-4

# Load the CSV file
csv_file = 'your_file.csv'  # Replace with your file path
data = pd.read_csv(csv_file)

# Extract the columns and convert from cm⁻¹ to eV
labels = data.iloc[:, 0]    # First column (labels)
y_cm = data.iloc[:, 1]      # Second column
x_cm = data.iloc[:, 2]      # Third column

y_eV = y_cm * hc_e_conversion_factor
x_eV = x_cm * hc_e_conversion_factor

# Get the column names for labels
y_label = data.columns[1] + ' (eV)'
x_label = data.columns[2] + ' (eV)'

# Generate markers (a, b, c, ...)
markers = list(string.ascii_lowercase[:len(labels)])

# Create the scatter plot
plt.scatter(x_eV, y_eV)

# Annotate each point with the corresponding letter marker
for i, marker in enumerate(markers):
    plt.annotate(marker, (x_eV[i], y_eV[i]), textcoords="offset points", xytext=(0, -10), ha='center')

# Set the aspect ratio to be equal and the limits for the axes
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title('Scatter Plot of Third Column vs Second Column (in eV)')
plt.show()

# Print the index relating markers to the original labels
print("Index:")
for marker, label in zip(markers, labels):
    print(f"{marker}: {label}")
```
#
```
#!/bin/bash

ev2cmi=8065.544645854528

# Function to process logs in a given directory
process_logs() {
    local dir=$1
    for f in "$dir"/*.log; do
        T1=$(grep 'Excited State ' "$f" | grep Triplet | head -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        T2=$(grep 'Excited State ' "$f" | grep Triplet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S1=$(grep 'Excited State ' "$f" | grep Singlet | head -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S2=$(grep 'Excited State ' "$f" | grep Singlet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S3=$(grep 'Excited State ' "$f" | grep Singlet | head -3 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S4=$(grep 'Excited State ' "$f" | grep Singlet | head -4 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S2_2T1=$(echo $S2 $T1 | awk '{print $1-2*$2}')
        S2_2T2=$(echo $S2 $T2 | awk '{print $1-2*$2}')
        S3_2T1=$(echo $S3 $T1 | awk '{print $1-2*$2}')
        echo "$dir $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2 $S3_2T1"
    done
}

# Process all relevant folders
for folder in */; do
    for method in CAMB3LYP Lc-wHPBE; do
        if [ -d "$folder/$method/tddft" ]; then
            process_logs "$folder/$method/tddft"
        fi
        if [ -d "$folder/$method/tda" ]; then
            process_logs "$folder/$method/tda"
        fi
    done
done
```
#
```
  echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDDFT/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -1 |           awk '{printf "%7i\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -2 | tail -1 | awk '{printf "%7i\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -7 | tail -1 | awk '{printf "%7i\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -8 | tail -1 | awk '{printf "%7i\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7i\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7i\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
echo "--------------"
for f in tda/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7i\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7i\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done
```
#
```
#!/bin/bash

ev2cmi=8065.544645854528

# Function to process logs in a given directory
process_logs() {
    local dir=$1
    for f in "$dir"/*.log; do
        T1=$(grep 'Excited State ' "$f" | grep Triplet | head -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        T2=$(grep 'Excited State ' "$f" | grep Triplet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S1=$(grep 'Excited State ' "$f" | grep Singlet | head -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S2=$(grep 'Excited State ' "$f" | grep Singlet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S3=$(grep 'Excited State ' "$f" | grep Singlet | head -3 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S4=$(grep 'Excited State ' "$f" | grep Singlet | head -4 | tail -1 | awk -v SF="$ev2cmi" '{print $5*SF}')
        S2_2T1=$(echo $S2 $T1 | awk '{print $1-2*$2}')
        S2_2T2=$(echo $S2 $T2 | awk '{print $1-2*$2}')
        S3_2T1=$(echo $S3 $T1 | awk '{print $1-2*$2}')
        echo "$dir $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2 $S3_2T1"
    done
}

# Process all relevant folders
for folder in */; do
    for method in CAMB3LYP Lc-wHPBE; do
        if [ -d "$folder/$method/tddft" ]; then
            process_logs "$folder/$method/tddft"
        fi
        if [ -d "$folder/$method/tda" ]; then
            process_logs "$folder/$method/tda"
        fi
    done
done
```
#
```
import os

def create_folders_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    # Open and read the file
    with open(file_path, 'r') as file:
        folder_names = file.readlines()

    # Create folders for each name in the file
    for folder_name in folder_names:
        folder_name = folder_name.strip()  # Remove any leading/trailing whitespace
        if folder_name:  # Ensure the folder name is not empty
            sanitized_folder_name = folder_name.replace(',', '_').replace('-', '_')
            try:
                os.makedirs(sanitized_folder_name, exist_ok=True)
                print(f"Created folder: {sanitized_folder_name}")
            except OSError as e:
                print(f"Error creating folder {sanitized_folder_name}: {e}")

# Specify the path to your text file
file_path = 'a.txt'

# Call the function to create folders
create_folders_from_file(file_path)
```
#
```
<svg width="400" height="180">
  <style>
    .column { width: 50px; height: 100px; }
    #col1 { fill: blue; }
    #col2 { fill: green; x: 60px; }
    #col3 { fill: red; x: 120px; }
  </style>
  <rect id="col1" class="column" x="0" y="10" />
  <rect id="col2" class="column" y="10" />
  <rect id="col3" class="column" y="10" />
</svg>
```
```
import numpy as np
import pandas as pd

# Function to calculate errors between two columns
def calculate_errors(data1, data2):
    mse = round(np.mean(data1 - data2), 3)
    mae = round(np.mean(np.abs(data1 - data2)), 3)
    sde = round(np.std(data1 - data2), 3)
    min_error = round(np.min(data1 - data2), 3)
    max_error = round(np.max(data1 - data2), 3)
    return mse, mae, sde, min_error, max_error

# Read the CSV files
csv_file1 = 'adc2_ccsdt_vtz.csv'  # Replace with your first CSV file path
csv_file2 = 'adc2_ccsd_vtz.csv'  # Replace with your second CSV file path

data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)

# Print the shape of data frames to debug
print(f"Shape of data1: {data1.shape}")
print(f"Shape of data2: {data2.shape}")

# Convert columns to numeric, forcing errors to NaN and then dropping NaNs
data1 = data1.apply(pd.to_numeric, errors='coerce')
data2 = data2.apply(pd.to_numeric, errors='coerce')

# Calculate errors for the specified column pairs
results = []
for col1, col2 in [(0, 0), (1, 1), (2, 2)]:  # Compare columns 1, 2, 3 (data1) with columns 1, 2, 3 (data2)
    if col1 < data1.shape[1] and col2 < data2.shape[1]:  # Ensure columns exist
        # Drop rows with NaN values in the relevant columns
        valid_indices = ~data1.iloc[:, col1].isna() & ~data2.iloc[:, col2].isna()
        if valid_indices.sum() > 0:  # Ensure there are valid data points
            mse, mae, sde, min_error, max_error = calculate_errors(
                data1.iloc[valid_indices, col1],
                data2.iloc[valid_indices, col2]
            )
            results.append((col1, col2, mse, mae, sde, min_error, max_error))
        else:
            print(f"No valid data points for columns {col1+1} (File 1) and {col2+1} (File 2)")
    else:
        print(f"Column {col1+1} or Column {col2+1} is out of bounds")

# Print results
for result in results:
    col1, col2, mse, mae, sde, min_error, max_error = result
    print(f"Column {col1+1} (File 1) vs Column {col2+1} (File 2)")  # Adjusted for 0-indexing
    print(f"Mean Signed Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Standard Deviation Error (SDE): {sde}")
    print(f"Min Error: {min_error}")
    print(f"Max Error: {max_error}")
    print()
```
#
```
f=SCS-PBE-QIDH/TDDFT/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -1 |           awk '{printf "%7i\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -2 | tail -1 | awk '{printf "%7i\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -7 | tail -1 | awk '{printf "%7i\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -8 | tail -1 | awk '{printf "%7i\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7i\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7i\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
echo "--------------"
for f in tda/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk -v SF="$ev2cmi" '{printf "%7i\n", $5*SF}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7i\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7i\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(csv_file, output_pdf):
    # Read CSV file with header
    df = pd.read_csv(csv_file)
    
    # Check if there are at least 4 columns
    if df.shape[1] < 4:
        raise ValueError("CSV file must have at least 4 columns.")

    # Create a 2x2 grid of histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot histograms for the first 4 columns
    columns = df.columns[:4]
    for i, col in enumerate(columns):
        ax = axs[i // 2, i % 2]
        ax.hist(df[col], bins=20, edgecolor='black')
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    # Adjust layout and save figure as PDF
    plt.tight_layout()
    plt.savefig(output_pdf)

# Example usage
csv_file = 'data.csv'
output_pdf = 'histograms.pdf'
plot_histograms(csv_file, output_pdf)
```
#
```
import csv

def sum_columns(csv_file_path):
    # Initialize a list to store the sum of each column
    column_sums = [0] * 17
    
    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        
        # Skip the header row
        header = next(reader)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Add the value of each column to the corresponding column sum
            for i in range(17):
                column_sums[i] += float(row[i])
    
    # Print the sum of each column
    for i, sum_value in enumerate(column_sums):
        print(f"Sum of column {i+1} ({header[i]}): {sum_value}")

# Example usage
csv_file_path = 'your_csv_file.csv'  # Replace with the path to your CSV file
sum_columns(csv_file_path)
```
#
```
import pandas as pd

# Load the CSV file
file_path = 'input.csv'
data = pd.read_csv(file_path)

# Separate rows based on the 'stability' column
stable_rows = data[data['stability'] == 0]
unstable_rows = data[data['stability'] == 1]

# Save the separated rows into different CSV files
stable_file_path = 's.csv'
unstable_file_path = 'us.csv'
stable_rows.to_csv(stable_file_path, index=False)
unstable_rows.to_csv(unstable_file_path, index=False)

print(f"Stable rows saved to {stable_file_path}")
print(f"Unstable rows saved to {unstable_file_path}")
```
#
```
import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file without headers
    df = pd.read_csv(input_file, header=None)

    # Perform the calculation: column 3 - (2 * column 9)
    df['calculation'] = df[2] - (2 * df[8])
    df['calculation'] = df['calculation'].round(3)  # Round to 3 decimal places

    # Assign 0 or 1 based on the result
    df['score'] = df['calculation'].apply(lambda x: 0 if x < 0 else 1)

    # Print the 2nd column (index 1) and calculation values of rows with score 1
    filtered_df = df[df['score'] == 1]
    for index, row in filtered_df.iterrows():
        print(f"{row[1]} {row['calculation']}")

    # Write only the 'score' column to the output CSV file
    df[['score']].to_csv(output_file, header=False, index=False)

# Example usage:
input_csv_file = 'tda_results.csv'
output_csv_file = 'score_tda.csv'
process_csv(input_csv_file, output_csv_file)
```
```
import csv

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        scores = []

        for row in reader:
            col3 = float(row[2])
            col9 = float(row[8])
            result = col3 - (2 * col9)
            score = 0 if result < 0 else 1
            scores.append([score])

    with open(output_file, 'w', newline='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(scores)

# Example usage:
input_csv_file = 'input.csv'
output_csv_file = 'output.csv'
process_csv(input_csv_file, output_csv_file)
```
```
import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file without headers
    df = pd.read_csv(input_file, header=None)

    # Perform the calculation: column 3 - (2 * column 9)
    df['calculation'] = df[2] - (2 * df[8])
    df['calculation'] = df['calculation'].round(3)  # Round to 3 decimal places

    # Assign 0 or 1 based on the result
    df['score'] = df['calculation'].apply(lambda x: 0 if x < 0 else 1)

    # Print the 2nd column (index 1) and calculation values of rows with score 1
    filtered_df = df[df['score'] == 1]
    for index, row in filtered_df.iterrows():
        print(f"{row[1]} {row['calculation']}")

    # Write only the 'score' column to the output CSV file
    df[['score']].to_csv(output_file, header=False, index=False)

# Example usage:
input_csv_file = 'tda_results.csv'
output_csv_file = 'score_tda.csv'
process_csv(input_csv_file, output_csv_file)
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

hartree2kcm = 627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('your_new_data_file.csv')

# Extract x and z values
x = data.iloc[:, 0]
z = data.iloc[:, 1]

# Convert energy values
z = z - np.min(z)
z = z * hartree2kcm

# Create a grid for x and y
grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(x.min(), x.max(), 100))

# Interpolating the data to fit the grid
grid_z = griddata((x, x), z, (grid_x, grid_y), method='cubic', fill_value=np.nan)

# Create a colormap with transparency
cmap = plt.cm.viridis
alpha = 0.4
cmap_colors = cmap(np.arange(cmap.N))
cmap_colors[:, -1] = alpha
cmap_alpha = ListedColormap(cmap_colors)

# Define contour levels
levels = np.linspace(-1, 30, 30)

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

cp = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap_alpha)
plt.colorbar(cp)

contour = plt.contour(grid_x, grid_y, grid_z, levels=levels, colors='black', linewidths=0.5)

plt.title('Contour Plot')
plt.xlabel("Values")
plt.ylabel("Values")
plt.savefig('contour_plot.png')
plt.show()
```
#
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hartree2kcm = 627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_MP2_new.csv')

# Extract x and z values
x = data.iloc[:, 0]
z = data.iloc[:, 1]

# Convert energy values
z = z - np.min(z)
z = z * hartree2kcm

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

plt.plot(x, z, label='Energy')
plt.title('Energy Plot')
plt.xlabel("Values")
plt.ylabel("Energy [$\Delta E$ in kcal/mol]")
plt.legend()
plt.savefig('energy_plot_new.png')
plt.show()
```
#
```
import os

def find_folders_with_mol(root_directory, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Walk through the directory
        for root, dirs, _ in os.walk(root_directory):
            # Iterate through the directories
            for dir_name in dirs:
                # Check if "Mol" is in the folder name
                if "Mol" in dir_name:
                    # Write the folder name to the output file
                    file.write(f"{dir_name}\n")

# Set the root directory to search and the output file path
root_directory = "your_directory_path_here"
output_file = "folders_with_mol.txt"

# Call the function
find_folders_with_mol(root_directory, output_file)
```
#
```
import os

# Define the template folder and file
template_folder = '/home/atreyee/Project_SOJT_Azaphenalenes/1APD3h_Mol_001/Contourplot_MP2/template'  # Change to your actual template folder path
template_file = 'opt.com'
output_base_folder = '/home/atreyee/Project_SOJT_Azaphenalenes/1APD3h_Mol_001/Contourplot_MP2'  # Change to your desired output folder path

# Define the range and increment for var1
var1_start = 1.3
var1_end = 1.5
var1_increment = 0.01

# Read the template file
with open(os.path.join(template_folder, template_file), 'r') as file:
    template_content = file.read()

# Create values for var1
var1_values = [round(var1_start + i * var1_increment, 2) for i in range(int((var1_end - var1_start) / var1_increment) + 1)]
var1_values.append(var1_end)  # Ensure the end value is included

# Remove duplicates and sort the values
var1_values = sorted(set(var1_values))

# Generate directories and files for each value of var1
for var1 in var1_values:
    # Create the directory name
    folder_name = f'Mol_{var1:.2f}'
    folder_path = os.path.join(output_base_folder, folder_name)

    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Replace the variable with the actual value
    modified_content = template_content.replace('var1', str(var1))

    # Write the modified content to a new opt.com file in the new directory
    new_file_path = os.path.join(folder_path, template_file)
    with open(new_file_path, 'w') as new_file:
        new_file.write(modified_content)

print("All combinations have been created.")
```
```
import pandas as pd
import matplotlib.pyplot as plt

# Constants
HARTREE_TO_KCAL_MOL = 627.509

# Load the CSV file with header
file_path = 'your_file.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Extract columns using header names
x = data.iloc[:, 0]  # First column
y_hartree = data.iloc[:, 1]  # Second column

# Convert from Hartree to kcal/mol
y_kcal_mol = y_hartree * HARTREE_TO_KCAL_MOL

# Plot the data
plt.plot(x, y_kcal_mol, label='Energy (kcal/mol)')
plt.xlabel(data.columns[0])  # Use the header of the first column as label
plt.ylabel('Energy (kcal/mol)')
plt.title('Energy Plot')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # Set square aspect ratio
plt.show()
```
# plot and conversion 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Conversion factor from Hartree to kcal/mol
hartree2kcm = 627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_MP2.csv', header=None)

# Extract x and z values
x = data.iloc[:, 0]
z = data.iloc[:, 1]

# Convert energy values
z = z - np.min(z)  # Subtract the minimum value from z
z = z * hartree2kcm  # Convert to kcal/mol

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, z, label='Energy')
ax.set_title('Energy Plot')
ax.set_xlabel("Values")
ax.set_ylabel("Energy [$\\Delta E$ in kcal/mol]")
ax.legend()

# Save and show the plot
plt.savefig('energy_plot.png')
plt.show()
```
# plot csv
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the specified columns with respect to the 1st column
plt.plot(df[0], df[1], label='Column 2', color='blue')
plt.plot(df[0], df[3], label='Column 4', color='green')
plt.plot(df[0], df[5], label='Column 6', color='red')

plt.title('Plot of Columns 2, 4, 6 vs 1st Column')
plt.xlabel('1st Column')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
```
#
```
import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file without headers
    df = pd.read_csv(input_file, header=None)

    # Perform the calculation: column 3 - (2 * column 9)
    df['calculation'] = df[2] - (2 * df[8])
    df['calculation'] = df['calculation'].round(3)  # Round to 3 decimal places

    # Assign 0 or 1 based on the result
    df['score'] = df['calculation'].apply(lambda x: 0 if x < 0 else 1)

    # Print the 2nd column (index 1) and calculation values of rows with score 1
    filtered_df = df[df['score'] == 1]
    for index, row in filtered_df.iterrows():
        print(f"{row[1]} {row['calculation']}")

    # Write only the 'score' column to the output CSV file
    df[['score']].to_csv(output_file, header=False, index=False)

# Example usage:
input_csv_file = 'tda_results.csv'
output_csv_file = 'score_tda.csv'
process_csv(input_csv_file, output_csv_file)
```
# 
```
import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file without headers
    df = pd.read_csv(input_file, header=None)

    # Perform the calculation: column 3 - (2 * column 9)
    df['score'] = df[2] - (2 * df[8])

    # Assign 0 or 1 based on the result
    df['score'] = df['score'].apply(lambda x: 0 if x < 0 else 1)

    # Print the 2nd column (index 1) of rows with score 1
    print(df[df['score'] == 1][1].tolist())

    # Write only the 'score' column to the output CSV file
    df[['score']].to_csv(output_file, header=False, index=False)

# Example usage:
input_csv_file = 'tda_results.csv'
output_csv_file = 'score_tda.csv'
process_csv(input_csv_file, output_csv_file)
```
#
```
import csv

def sum_columns(csv_file_path):
    # Initialize a list to store the sum of each column
    column_sums = [0] * 17
    
    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        
        # Skip the header row
        header = next(reader)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Add the value of each column to the corresponding column sum
            for i in range(17):
                column_sums[i] += float(row[i])
    
    # Print the sum of each column
    for i, sum_value in enumerate(column_sums):
        print(f"Sum of column {i+1} ({header[i]}): {sum_value}")

# Example usage
csv_file_path = 'your_csv_file.csv'  # Replace with the path to your CSV file
sum_columns(csv_file_path)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file
file1 = "file1.csv"
df1 = pd.read_csv(file1)

# Read the second CSV file
file2 = "file2.csv"
df2 = pd.read_csv(file2)

# Extract the 6th, 8th, and 9th columns from both files
data1 = df1.iloc[:, [5, 7, 8]]  # 6th, 8th, and 9th columns
data2 = df2.iloc[:, [5, 7, 8]]  # 6th, 8th, and 9th columns

# Filter rows where the 8th column has the value "CS"
cs_data1 = data1[data1.iloc[:, 1] == "CS"]
cs_data2 = data2[data2.iloc[:, 1] == "CS"]

# Plot scatter plot for non-CS points
plt.scatter(data1.iloc[:, 0], data1.iloc[:, 2], color='blue', label='File 1')
plt.scatter(data2.iloc[:, 0], data2.iloc[:, 2], color='green', label='File 2')

# Plot scatter plot for CS points
plt.scatter(cs_data1.iloc[:, 0], cs_data1.iloc[:, 2], color='red', label='File 1 (CS)')
plt.scatter(cs_data2.iloc[:, 0], cs_data2.iloc[:, 2], color='red', label='File 2 (CS)')

# Add labels and title
plt.xlabel('6th Column')
plt.ylabel('9th Column')
plt.title('Scatter Plot with "CS" Points Highlighted')

# Add legend
plt.legend()

# Show plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file
file1 = "file1.csv"
df1 = pd.read_csv(file1)

# Read the second CSV file
file2 = "file2.csv"
df2 = pd.read_csv(file2)

# Extract the 6th and 8th columns from both files
data1 = df1.iloc[:, [5, 7]]  # 6th and 8th columns
data2 = df2.iloc[:, [5, 7]]  # 6th and 8th columns

# Filter rows where the 8th column has the value "CS"
cs_data1 = data1[data1.iloc[:, 1] == "CS"]
cs_data2 = data2[data2.iloc[:, 1] == "CS"]

# Plot scatter plot
plt.scatter(cs_data1.iloc[:, 0], cs_data1.iloc[:, 1], color='blue', label='File 1 (CS)')
plt.scatter(cs_data2.iloc[:, 0], cs_data2.iloc[:, 1], color='red', label='File 2 (CS)')

# Add labels and title
plt.xlabel('6th Column')
plt.ylabel('8th Column')
plt.title('Scatter Plot with "CS" Points Highlighted')

# Add legend
plt.legend()

# Show plot
plt.show()
```
#
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load the CSV file
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Extract the data
var1 = data['var1'].values
energies_hartree = data['energies'].values

# Convert energies from hartree to kcal/mol
hartree2kcm = 627.509
energies_kcal = energies_hartree * hartree2kcm

# Create grid data for contour plot
xi = np.linspace(var1.min(), var1.max(), 100)
yi = np.linspace(energies_kcal.min(), energies_kcal.max(), 100)
zi = griddata((var1, energies_kcal), energies_kcal, (xi[None, :], yi[:, None]), method='cubic')

# Plotting the contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
plt.colorbar(contour)

# Adding labels and title
plt.xlabel('var1')
plt.ylabel('energies (kcal/mol)')
plt.title('Contour Plot of Energies (kcal/mol) vs Var1')

# Save the plot as a PDF file
output_pdf_path = 'contour_plot.pdf'
plt.savefig(output_pdf_path, format='pdf')

# Show the plot
plt.show()
```
#
```
import numpy as np
import pandas as pd

# Function to calculate errors between two columns
def calculate_errors(data1, data2):
    mse = round(np.mean(data1 - data2), 3)
    mae = round(np.mean(np.abs(data1 - data2)), 3)
    sde = round(np.std(data1 - data2), 3)
    min_error = round(np.min(data1 - data2), 3)
    max_error = round(np.max(data1 - data2), 3)
    return mse, mae, sde, min_error, max_error

# Read the CSV files
csv_file1 = 'adc2_ccsdt_vtz.csv'  # Replace with your first CSV file path
csv_file2 = 'adc2_ccsd_vtz.csv'  # Replace with your second CSV file path

data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)

# Print the shape of data frames to debug
print(f"Shape of data1: {data1.shape}")
print(f"Shape of data2: {data2.shape}")

# Convert columns to numeric, forcing errors to NaN and then dropping NaNs
data1 = data1.apply(pd.to_numeric, errors='coerce')
data2 = data2.apply(pd.to_numeric, errors='coerce')

# Calculate errors for the specified column pairs
results = []
for col1, col2 in [(0, 0), (1, 1), (2, 2)]:  # Compare columns 1, 2, 3 (data1) with columns 1, 2, 3 (data2)
    if col1 < data1.shape[1] and col2 < data2.shape[1]:  # Ensure columns exist
        # Drop rows with NaN values in the relevant columns
        valid_indices = ~data1.iloc[:, col1].isna() & ~data2.iloc[:, col2].isna()
        if valid_indices.sum() > 0:  # Ensure there are valid data points
            mse, mae, sde, min_error, max_error = calculate_errors(
                data1.iloc[valid_indices, col1],
                data2.iloc[valid_indices, col2]
            )
            results.append((col1, col2, mse, mae, sde, min_error, max_error))
        else:
            print(f"No valid data points for columns {col1+1} (File 1) and {col2+1} (File 2)")
    else:
        print(f"Column {col1+1} or Column {col2+1} is out of bounds")

# Print results
for result in results:
    col1, col2, mse, mae, sde, min_error, max_error = result
    print(f"Column {col1+1} (File 1) vs Column {col2+1} (File 2)")  # Adjusted for 0-indexing
    print(f"Mean Signed Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Standard Deviation Error (SDE): {sde}")
    print(f"Min Error: {min_error}")
    print(f"Max Error: {max_error}")
    print()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

def plot_selected_columns(csv_file):
    try:
        # Read the CSV file into a DataFrame, assuming the first row is the header
        # Use the 'on_bad_lines' parameter to skip lines with too many fields
        df = pd.read_csv(csv_file, header=0, on_bad_lines='skip')

        # Check if the required columns (1st, 3rd, 4th, and 5th) exist
        if df.shape[1] < 5:
            raise ValueError("The CSV file must contain at least 5 columns.")

        # Generate x-axis values from the first column
        x = df.iloc[:, 0].astype(float).to_numpy()

        # Columns to plot: 3rd, 4th, and 5th (index 2, 3, and 4)
        columns_to_plot = [2, 3, 4]
        labels = ['col3', 'col4', 'col5']

        for col_index, label in zip(columns_to_plot, labels):
            try:
                # Convert the column to numeric values
                column = df.iloc[:, col_index].astype(float).to_numpy()
                # Plot the data
                plt.plot(x, column, label=label)
            except ValueError:
                print(f"Skipping non-numeric column: {df.columns[col_index]}")

        plt.xlabel('Displacement [Å]')
        plt.ylabel('E')
        plt.title('Plot of Selected Columns from CSV File')
        plt.legend()
        plt.show()

    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")

# Usage example
csv_file = 'Mol89_energies_wB97XD3_0.05.csv'  # Replace with the path to your CSV file
plot_selected_columns(csv_file)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the specified columns with respect to the 1st column
plt.plot(df[0], df[1], label='Column 2', color='blue')
plt.plot(df[0], df[3], label='Column 4', color='green')
plt.plot(df[0], df[5], label='Column 6', color='red')

plt.title('Plot of Columns 2, 4, 6 vs 1st Column')
plt.xlabel('1st Column')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the 2nd column (index 1)
plt.plot(df[1])
plt.title('Plot of the 2nd Column')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```
# DNC extract
```
#!/bin/bash

# Define the output file
output_file="output.txt"

# Loop through all trajectory blocks in the output file
for ((i=1; i<=41; i++)); do
    echo "Trajectory $i:"
    # Use grep and awk to extract the value under "DNC"
    value=$(grep -A 5 "Trajectory\s*$i" "$output_file" | awk '/DNC/{getline; print $1}')
    echo "Value under DNC: $value"
    echo "---------------------"
done
```
# Assignment 
```
import numpy as np
import matplotlib.pyplot as plt
import csv

# Define the time periods
days_1 = np.arange(1, 29)  # Days 1 to 28
days_2 = np.arange(29, 57)  # Days 29 to 56

# Define the weight functions
def W1(t):
    return 48 + 3.64*t + 0.6363*t**2 + 0.00963*t**3

def W2(t):
    return -1004 + 65.8*t

# Calculate weights for both periods
weights_1 = W1(days_1)
weights_2 = W2(days_2)

# Combine the days and weights into one array
days_total = np.concatenate((days_1, days_2))
weights_total = np.concatenate((weights_1, weights_2))

# Plot the weight over the first 56 days
plt.figure(figsize=(10, 6))
plt.plot(days_1, weights_1, label='W1(t) for Days 1-28', color='blue')
plt.plot(days_2, weights_2, label='W2(t) for Days 29-56', color='red')
plt.xlabel('Days')
plt.ylabel('Weight (grams)')
plt.title('Bird Weight Over 56 Days')
plt.legend()
plt.grid(True)
plt.show()

# Prepare the CSV data
csv_data = np.column_stack((days_total, weights_total))

# Save the CSV file
csv_file_path = 'bird_weight_over_56_days.csv'
header = ['Day', 'Weight']

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    writer.writerows(csv_data)

print(f"CSV file saved as {csv_file_path}")
```
#
```
import math

def calculate_expression(k1, k2, k3, E12, E13, kB, T):
    numerator = k1 + k2 * math.exp(E12 / (kB * T)) + k3 * math.exp(E13 / (kB * T))
    denominator = 1 + math.exp(E12 / (kB * T)) + math.exp(E13 / (kB * T))
    result = numerator / denominator
    return result

# Example usage:
k1 = 1
k2 = 2
k3 = 3
E12 = 10
E13 = 20
kB = 1.38e-23  # Boltzmann constant in Joules per Kelvin
T = 300        # Temperature in Kelvin

result = calculate_expression(k1, k2, k3, E12, E13, kB, T)
print("Result:", result)
```
#
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
#
```
import csv

def sort_by_difference(input_file):
    # Read the contents of the CSV file
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header
        data = list(reader)

    # Calculate the difference between column 6 and column 5 and store it along with the corresponding row
    diff_data = []
    for row in data:
        col_5 = float(row[5])
        col_6 = float(row[6])
        difference = col_6 - col_5
        diff_data.append((difference, row))

    # Sort the data based on the calculated difference
    sorted_diff_data = sorted(diff_data, key=lambda x: x[0])

    # Write the sorted data to a new CSV file
    with open('sorted_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write the header
        for _, row in sorted_diff_data:
            writer.writerow(row)

if __name__ == "__main__":
    input_file = 'original.csv'  # Specify the CSV file
    sort_by_difference(input_file)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the 2nd column (index 1)
plt.plot(df[1])
plt.title('Plot of the 2nd Column')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```
#
```
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('your_file.csv')

# Group rows based on the values in the 5th column
grouped_data = df.groupby(df.iloc[:, 4])

# Initialize an empty dictionary to store grouped rows
grouped_rows = {}

# Iterate over each group
for group_name, group_data in grouped_data:
    # Store the rows of each group in the dictionary
    grouped_rows[group_name] = group_data.values.tolist()

# Print the grouped rows
for key, value in grouped_rows.items():
    print(f"Rows with value {key} in the 5th column:")
    for row in value:
        print(row)
    print()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file
file1 = "file1.csv"
df1 = pd.read_csv(file1)

# Read the second CSV file
file2 = "file2.csv"
df2 = pd.read_csv(file2)

# Extract the 6th, 8th, and 9th columns from both files
data1 = df1.iloc[:, [5, 7, 8]]  # 6th, 8th, and 9th columns
data2 = df2.iloc[:, [5, 7, 8]]  # 6th, 8th, and 9th columns

# Filter rows where the 8th column has the value "CS"
cs_data1 = data1[data1.iloc[:, 1] == "CS"]
cs_data2 = data2[data2.iloc[:, 1] == "CS"]

# Plot scatter plot for non-CS points
plt.scatter(data1.iloc[:, 0], data1.iloc[:, 2], color='blue', label='File 1')
plt.scatter(data2.iloc[:, 0], data2.iloc[:, 2], color='green', label='File 2')

# Plot scatter plot for CS points
plt.scatter(cs_data1.iloc[:, 0], cs_data1.iloc[:, 2], color='red', label='File 1 (CS)')
plt.scatter(cs_data2.iloc[:, 0], cs_data2.iloc[:, 2], color='red', label='File 2 (CS)')

# Add labels and title
plt.xlabel('6th Column')
plt.ylabel('9th Column')
plt.title('Scatter Plot with "CS" Points Highlighted')

# Add legend
plt.legend()

# Show plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files without headers
csv1 = 'tda_results.csv'
csv2 = 'tddft_results.csv'

df1 = pd.read_csv(csv1, header=None)
df2 = pd.read_csv(csv2, header=None)

# Filter the rows based on the given ranges
filtered_df1 = df1[(df1.iloc[:, 8] >= 1.5) & (df1.iloc[:, 8] <= 4.0)]
filtered_df2 = df2[(df2.iloc[:, 8] >= 0.0) & (df2.iloc[:, 8] <= 2.0)]

# Find common indices in both filtered DataFrames
common_indices = filtered_df1.index.intersection(filtered_df2.index)

# Filter the DataFrames again to only include common indices
filtered_df1 = filtered_df1.loc[common_indices]
filtered_df2 = filtered_df2.loc[common_indices]

# Plot a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(filtered_df1.iloc[:, 8], filtered_df2.iloc[:, 8], c='blue', label='Filtered Data')
plt.xlabel('tda')
plt.ylabel('tddft')
plt.title('Scatter Plot')
plt.legend()
plt.grid(True)
plt.axis('square')
plt.show()

# Prepare the data for output
tda_column1 = filtered_df1.iloc[:, 0].reset_index(drop=True)
tda_column2 = filtered_df1.iloc[:, 1].reset_index(drop=True)
tda_column8 = filtered_df1.iloc[:, 8].astype(str).reset_index(drop=True)
tddft_column8 = filtered_df2.iloc[:, 8].astype(str).reset_index(drop=True)

# Combine columns with underscore
combined_column8 = tda_column8 + '_' + tddft_column8

# Create DataFrame for output
combined_data = pd.DataFrame({
    'Column1': tda_column1,
    'Column2': tda_column2,
    'CombinedColumn8': combined_column8
})

# Save to .smi file
combined_data.to_csv('combined_results.smi', sep='\t', index=False, header=False)

print("Data has been saved to 'combined_results.smi'.")
```
#
```
# Defining the sets as dictionaries for easier access
A = [1.41684998, 1.39586138, 118.73213541, 1.39808826, 118.14012765, 2.75392309, 1.35596387, 121.01850011, 1.32328927, 115.31283627, 1.33454246, 119.70103198, 1.07941933, 116.93824030, 3.83689884, 1.08479019, 115.90926862]
B = [1.41618473, 1.39548941, 118.71728125, 1.39773781, 118.12533163, 2.75274937, 1.35539209, 121.01626820, 1.32285087, 115.29290424, 1.33412700, 119.71276677, 1.07921391, 116.95899147, 3.83555674, 1.08462730, 115.90036750]
C = [1.41777170, 1.39703257, 118.72307975, 1.39916112, 118.12374420, 2.75582180, 1.35686210, 121.01448739, 1.32436910, 115.29555532, 1.33558725, 119.69745749, 1.08031159, 116.94254615, 3.83973383, 1.08575020, 115.90919061]
D = [1.42171145, 1.40060982, 118.70025962, 1.40371658, 118.09471826, 2.76440173, 1.36051130, 121.03019348, 1.32747581, 115.25443118, 1.33903290, 119.71820111, 1.08370603, 116.95231168, 3.85209159, 1.08924983, 115.86593614]
E = [1.42191145, 1.40100982, 118.69545962, 1.40331658, 118.07521826, 2.76370173, 1.35951130, 121.09239348, 1.32707581, 115.34623118, 1.33823290, 119.66590111, 1.08320603, 116.99401168, 3.85069159, 1.08834983, 115.91013614]

# Finding the length of the longest set for formatting purposes
max_len = max(len(A), len(B), len(C), len(D), len(E))

# Print the sets in columns
print(f"{'A':<20} {'B':<20} {'C':<20} {'D':<20} {'E':<20}")
for i in range(max_len):
    a_val = f"{A[i]:<20}" if i < len(A) else " " * 20
    b_val = f"{B[i]:<20}" if i < len(B) else " " * 20
    c_val = f"{C[i]:<20}" if i < len(C) else " " * 20
    d_val = f"{D[i]:<20}" if i < len(D) else " " * 20
    e_val = f"{E[i]:<20}" if i < len(E) else " " * 20
    print(f"{a_val} {b_val} {c_val} {d_val} {e_val}")
```
#
```
import csv

# Open the input CSV file ('a.csv') and create the output CSV file ('c.csv')
with open('a.csv', 'r') as input_file, open('c.csv', 'w', newline='') as output_file:
    # Create CSV reader and writer objects
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Iterate over each row in the input CSV file
    for row in reader:
        # Extract the first column from each row and write it to the output CSV file
        writer.writerow([row[0]])
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return arr, elapsed_time

def measure_time(N):
    arr = np.random.rand(N) * 100
    # Sort the array and measure time taken
    _, elapsed_time = bubble_sort(arr)
    return elapsed_time

# Array sizes from 2^0 to 2^12
array_sizes = [2**i for i in range(13)]
times = []

# Time taken for each array size
for N in array_sizes:
    t = measure_time(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")
M=[]
for N in array_sizes:
    M.append((N**2-N)/2)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, times, marker='o', linestyle='--', color='b')
plt.plot(array_sizes,M, color='g')
plt.xlabel('Array Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Bubble Sort Time Complexity')
plt.grid(True)
plt.show()
```
```
import time

def bubble_sort(arr):
    
    start_time = time.time()

    n = len(arr)
    
    for i in range(0,n-1):
        for j in range(i+1,n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    return arr, elapsed_time
```
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return arr, elapsed_time

def measure_time(N):
     arr = np.random.rand(N) * 100
    # Sort the array and measure time taken
     start_time = time.time()
     sorted_array = bubble_sort(arr)
     end_time = time.time()
     time_bubble = end_time - start_time
     return time_bubble

def measure_time2(N):
     arr = np.random.rand(N) * 100
     start_time = time.time()
     sorted_array = np.sort(arr, kind='quicksort')
     end_time = time.time()
     time_quicksort = end_time - start_time
     return  time_quicksort

def measure_time3(N):
     arr = np.random.rand(N) * 100
     start_time = time.time()
     sorted_array = np.sort(arr, kind='mergesort')
     end_time = time.time()
     time_mergesort = end_time - start_time
     return time_mergesort

def measure_time4(N):
     arr = np.random.rand(N) * 100
     start_time = time.time()
     sorted_array = np.sort(arr, kind='heapsort')
     end_time = time.time()
     time_heapsort = end_time - start_time
     return time_heapsort

# Array sizes from 2^0 to 2^12
array_sizes = [2**i for i in range(13)]
times = []

# Time taken for each array size
print("bubblesort")
for N in array_sizes:
    t = measure_time(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

print("quicksort")
for N in array_sizes:
    t = measure_time2(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

print("mergesort")
for N in array_sizes:
    t = measure_time3(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

print("heapsort")
for N in array_sizes:
    t = measure_time4(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return arr, elapsed_time

def measure_time(N):
    arr = np.random.rand(N) * 100
    # Sort the array and measure time taken
    _, elapsed_time = bubble_sort(arr)
    return elapsed_time

# Array sizes from 2^0 to 2^12
array_sizes = [2**i for i in range(13)]
times = []

# Time taken for each array size
for N in array_sizes:
    t = measure_time(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, times, marker='o', linestyle='--', color='b')
plt.xlabel('Array Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Bubble Sort Time Complexity')
plt.grid(True)
plt.show()
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    start_time = time.time()
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Define the size range as powers of 2 from 2^0 to 2^13
sizes = [2**i for i in range(14)]

# Initialize lists to store times
times_bubble = []
times_quicksort = []
times_mergesort = []
times_heapsort = []

for size in sizes:
    array = np.random.rand(size)
    
    # Bubble Sort
    times_bubble.append(bubble_sort(array.copy()))
    
    # Quick Sort
    start_time = time.time()
    np.sort(array.copy(), kind='quicksort')
    end_time = time.time()
    times_quicksort.append(end_time - start_time)
    
    # Merge Sort
    start_time = time.time()
    np.sort(array.copy(), kind='mergesort')
    end_time = time.time()
    times_mergesort.append(end_time - start_time)
    
    # Heap Sort
    start_time = time.time()
    np.sort(array.copy(), kind='heapsort')
    end_time = time.time()
    times_heapsort.append(end_time - start_time)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the CPU times
plt.plot(sizes, times_bubble, label='Bubble Sort', marker='o')
plt.plot(sizes, times_quicksort, label='Quick Sort', marker='o')
plt.plot(sizes, times_mergesort, label='Merge Sort', marker='o')
plt.plot(sizes, times_heapsort, label='Heap Sort', marker='o')

# Plot aN^2 and bNlog(N)
a = 1e-7  # Adjustable constant
b = 1e-6  # Adjustable constant
plt.plot(sizes, [a*(n**2) for n in sizes], label='$aN^2$', linestyle='--')
plt.plot(sizes, [b*n*np.log(n) for n in sizes], label='$bN\log(N)$', linestyle='--')

# Labeling the plot
plt.xlabel('Array Size (N)')
plt.ylabel('CPU Time (s)')
plt.title('CPU Time for Various Sorting Algorithms')
plt.legend()
plt.grid(True)
plt.xscale('log', base=2)
plt.show()
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return arr, elapsed_time

def measure_time(N):
    # Generate a random array of size N with values between 0 and 100
    arr = np.random.rand(N) * 100
    # Sort the array and measure time taken
    _, elapsed_time = bubble_sort(arr)
    return elapsed_time

# Array sizes to test
array_sizes = [10, 100, 1000, 10000]
times = []

# Measure the time taken for each array size
for N in array_sizes:
    t = measure_time(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, times, marker='o', linestyle='--', color='b')
plt.xlabel('Array Size (N)')
plt.ylabel('Time Taken (seconds)')
plt.title('Bubble Sort Time Complexity')
plt.grid(True)
plt.show()

# Comment on the shape of the function
print("The shape of the function is expected to be quadratic since bubble sort has a time complexity of O(N^2). As N increases, the time taken grows significantly.")
```
#csv file modification
```
import csv

# Define the input and output CSV file paths
input_file = 'input.csv'
output_file = 'output.csv'

# Open the input CSV file for reading
with open(input_file, 'r') as infile:
    reader = csv.reader(infile)
    
    # Open the output CSV file for writing
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Skip the header in the input file
        header = next(reader)
        
        # Write the header for the new column
        writer.writerow(['New Column'])
        
        # Process each row
        for row in reader:
            # Check the value of the 3rd column (index 2)
            value = float(row[2])
            if value < 0.8:
                new_value = 1
            else:
                new_value = 0
            
            # Write the new value as a single-column row in the output file
            writer.writerow([new_value])
```
#
```
import numpy as np
import matplotlib.pyplot as plt

# Define the function for Cv as a function of y = kBT/hv
def Cv(y):
    return 3 * y**2 * np.exp(1/y) / (np.exp(1/y) - 1)**2

# Generate values for y
y_values = np.linspace(0.01, 10, 500)
Cv_values = Cv(y_values)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(y_values, Cv_values, label=r"$C_V$ as a function of $\frac{k_B T}{h \nu}$")
plt.xlabel(r"$\frac{k_B T}{h \nu}$", fontsize=14)
plt.ylabel(r"$C_V/Nk_B$", fontsize=14)
plt.title(r"Einstein's Heat Capacity Model", fontsize=16)
plt.axhline(y=3, color='red', linestyle='--', label="Dulong-Petit Law")
plt.legend()
plt.grid(True)
plt.show()
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    start_time = time.time()
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Define the size range as powers of 2 from 2^0 to 2^13
sizes = [2**i for i in range(14)]

# Initialize lists to store times
times_bubble = []
times_quicksort = []
times_mergesort = []
times_heapsort = []

for size in sizes:
    array = np.random.rand(size)
    
    # Bubble Sort
    times_bubble.append(bubble_sort(array.copy()))
    
    # Quick Sort
    start_time = time.time()
    np.sort(array.copy(), kind='quicksort')
    end_time = time.time()
    times_quicksort.append(end_time - start_time)
    
    # Merge Sort
    start_time = time.time()
    np.sort(array.copy(), kind='mergesort')
    end_time = time.time()
    times_mergesort.append(end_time - start_time)
    
    # Heap Sort
    start_time = time.time()
    np.sort(array.copy(), kind='heapsort')
    end_time = time.time()
    times_heapsort.append(end_time - start_time)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the CPU times
plt.plot(sizes, times_bubble, label='Bubble Sort', marker='o')
plt.plot(sizes, times_quicksort, label='Quick Sort', marker='o')
plt.plot(sizes, times_mergesort, label='Merge Sort', marker='o')
plt.plot(sizes, times_heapsort, label='Heap Sort', marker='o')

# Plot aN^2 and bNlog(N)
a = 1e-7  # Adjustable constant
b = 1e-6  # Adjustable constant
plt.plot(sizes, [a*(n**2) for n in sizes], label='$aN^2$', linestyle='--')
plt.plot(sizes, [b*n*np.log(n) for n in sizes], label='$bN\log(N)$', linestyle='--')

# Labeling the plot
plt.xlabel('Array Size (N)')
plt.ylabel('CPU Time (s)')
plt.title('CPU Time for Various Sorting Algorithms')
plt.legend()
plt.grid(True)
plt.xscale('log', base=2)
plt.show()
```
#
```
`import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return arr, elapsed_time

def measure_time(N):
    arr = np.random.rand(N) * 100
    # Sort the array and measure time taken
    _, elapsed_time = bubble_sort(arr)
    return elapsed_time

# Array sizes from 2^0 to 2^12
array_sizes = [2**i for i in range(13)]
times = []

# Time taken for each array size
for N in array_sizes:
    t = measure_time(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, times, marker='o', linestyle='--', color='b')
plt.xlabel('Array Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Bubble Sort Time Complexity')
plt.grid(True)
plt.show()
```
#
```import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file without headers
    df = pd.read_csv(input_file, header=None)

    # Perform the calculation: column 3 - (2 * column 9)
    df['score'] = df[2] - (2 * df[8])

    # Assign 0 or 1 based on the result
    df['score'] = df['score'].apply(lambda x: 0 if x < 0 else 1)

    # Print the 2nd column (index 1) of rows with score 1
    for value in df[df['score'] == 1][1]:
        print(value)

    # Write only the 'score' column to the output CSV file
    df[['score']].to_csv(output_file, header=False, index=False)

# Example usage:
input_csv_file = 'tda_results.csv'
output_csv_file = 'score_tda.csv'
process_csv(input_csv_file, output_csv_file)
```
#
```#!/bin/bash

# Define the root directory where the folders are located
root_dir="/path/to/root/directory"

# Function to extract numeric value from opt.out file
extract_numeric_value() {
    file="$1"
    numeric_value=$(grep -oP 'MP2/DEF2-SVP//MP2/DEF2-SVP energy=\s*\K-?\d+\.\d+' "$file")
    if [ -n "$numeric_value" ]; then
        echo "$numeric_value"
    fi
}

# Iterate through folders starting with "Mol"
for folder in "$root_dir"/Mol*/; do
    if [ -d "$folder" ]; then
        opt_out_file="$folder/opt.out"
        if [ -f "$opt_out_file" ]; then
            numeric_value=$(extract_numeric_value "$opt_out_file")
            echo "$numeric_value"
        fi
    fi
done
```
#
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
#
```
import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file paths
csv_files = ['data_MP2.csv', 'data_wB97XD3.csv', 'data_B3LYP.csv', 'data_PBE0.csv', 'data_PBE.csv']
labels = ['a', 'b', 'c', 'd', 'e']

# Initialize a plot
plt.figure(figsize=(10, 6))

# Store all the data in a list to determine a common zoom range
all_data = []

# Loop through each file and plot the density of the 6th column
for file, label in zip(csv_files, labels):
    df = pd.read_csv(file)
    if df.shape[1] > 5:  # Ensure the file has at least 6 columns
        data = df.iloc[:, 5]  # Extract the 6th column
        all_data.append(data)
        data.plot(kind='density', label=label)

# Combine all data to find the zoom range
combined_data = pd.concat(all_data)

# Manually set x-axis limits for zooming in (adjust these values based on your data)
plt.xlim(left=combined_data.min() - 1, right=combined_data.max() + 1)

# Customize the plot
plt.title('Density Plots of the 6th Column from Each CSV File')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()

# Save the plot as a PDF
plt.savefig('density_plot.pdf')

# Show the plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file paths
csv_files = ['data_MP2.csv', 'data_wB97XD3.csv', 'data_B3LYP.csv', 'data_PBE0.csv', 'data_PBE.csv']

# Initialize a plot
plt.figure(figsize=(10, 6))

# Store all the data in a list to determine a common zoom range
all_data = []

# Loop through each file and plot the density of the 6th column
for file in csv_files:
    df = pd.read_csv(file)
    if df.shape[1] > 5:  # Ensure the file has at least 6 columns
        data = df.iloc[:, 5]  # Extract the 6th column
        all_data.append(data)
        data.plot(kind='density', label=file)

# Combine all data to find the zoom range
combined_data = pd.concat(all_data)

# Calculate zoom range around the median and interquartile range
median = combined_data.median()
q1 = combined_data.quantile(0.25)
q3 = combined_data.quantile(0.75)
iqr = q3 - q1
zoom_range = (median - 1.5 * iqr, median + 1.5 * iqr)

# Customize the plot
plt.title('Density Plots of the 6th Column from Each CSV File')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()

# Set x-axis limits to zoom in
plt.xlim(zoom_range)

# Save the plot as a PDF
plt.savefig('density_plot.pdf')

# Show the plot
plt.show()
```
#
```
import csv
import numpy as np

def calculate_errors(data):
    # Extracting columns
    second_column = data[:, 1].astype(float)
    third_column = data[:, 2].astype(float)
    
    # Calculating errors
    mae_third = np.mean(np.abs(second_column - third_column))
    mse_third = np.mean((second_column - third_column)**2)
    std_third = np.std(second_column - third_column)
    
    return mae_third, mse_third, std_third

def main(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader))
    
    mae_third, mse_third, std_third = calculate_errors(data)
    
    print(f"MAE of the third column: {mae_third}")
    print(f"MSE of the third column: {mse_third}")
    print(f"Standard Deviation of the third column: {std_third}")

if __name__ == "__main__":
    csv_file_path = "data.csv"  # Replace with your CSV file path
    main(csv_file_path)
```
#
```
import re

def read_indices(file_path):
    """Reads indices from a file."""
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def backup_file(input_file, backup_file):
    """Creates a backup of the input file."""
    with open(input_file, 'r') as f:
        content = f.read()
    with open(backup_file, 'w') as f:
        f.write(content)

def replace_carbon_with_nitrogen(content, carbon_idx, hydrogen_atom=None):
    """Replaces carbon atom with nitrogen and removes associated hydrogen if applicable."""
    nitrogen_atom = f'N{carbon_idx[1:]}'  # Replace 'C' with 'N' but keep the index

    # Replace the carbon with nitrogen in the Z-matrix
    content = re.sub(rf'\b{carbon_idx}\b', nitrogen_atom, content)

    # If a hydrogen atom is associated, remove its lines
    if hydrogen_atom:
        content = re.sub(rf'\n.*{hydrogen_atom}.*', '', content)

    return content

def main():
    input_file = '1AP_c2v.com'
    output_file = 'opt.com'
    indices_file = 'indscr.txt'
    hydrogen_mapping = {
        'C6': 'H15',
        'C7': 'H16',
        'C8': 'H17',
        'C9': 'H18',
        'C10': 'H19',
        'C11': 'H20',
        'C12': 'H21',
        'C13': 'H22',
        'C14': 'H23'
    }

    # Read indices and original input file
    indices = read_indices(indices_file)
    
    # Read the original geometry from the input file
    with open(input_file, 'r') as f:
        content = f.read()

    # Process each index (replace carbon with nitrogen and remove associated hydrogen)
    for index in indices:
        carbon_atom = f'C{index}'
        hydrogen_atom = hydrogen_mapping.get(carbon_atom)

        # Create a backup before making changes
        backup_file(input_file, 'backup_inp_scr.com')

        # Replace carbon with nitrogen and remove hydrogens
        content = replace_carbon_with_nitrogen(content, carbon_atom, hydrogen_atom)

    # Write the modified content to the output file
    with open(output_file, 'w') as f:
        f.write(content)

    print(f'Processed file saved as {output_file}')

if __name__ == '__main__':
    main()
```
#
```
#!/bin/bash

# Define directories to process
directories=("1_2_3_4_5_6_7_9_octaaza_C2v" 
             "1_2_3_4_6_7_9_heptaaza_C2v" 
             "1_2_3_6_7_pentaaza_C2v" 
             "1_2_5_6_8_pentaaza_C2v" 
             "1_3_4_6_8_pentaaza_C2v" 
             "1_3_6_7_tetraaza_C2v" 
             "1_6_biaza_C2v")

# Loop over directories
for dir in "${directories[@]}"
do
    echo "Processing folder: $dir"

    # Copy template file to the directory
    cp 1AP_c2v.com $dir/1AP_c2v.com

    # Move to the directory
    cd $dir

    # Run the Python script to generate the input file
    python3 ../make_inp.py

    # Rename the generated opt.com file for clarity
    mv opt.com "${dir}_opt.com"

    # Go back to the parent directory
    cd ..

    echo "Processed folder: $dir"
done

echo "All folders processed!"
```
#
```
#!/bin/bash

# Ensure the script is executable
chmod +x make_inp.py

# Read the number of molecules from indices.txt
Nmols=$(wc -l < indices.txt)

for imol in $(seq $Nmols); do
  # Get the folder name from folders.txt
  folder=$(sed -n "${imol}p" folders.txt)
  
  # Extract the current index from indices.txt
  sed -n "${imol}p" indices.txt > indscr.txt
  
  # Create the folder if it doesn't exist
  mkdir -p "../$folder"
  
  # Copy the template file and run the Python script
  cp 1AP_c2v.com inp_scr.com
  python3 make_inp.py

  # Move the generated opt.com to the new folder
  mv opt.com "../$folder"

  echo "Processed folder: $folder"
done

echo "All folders processed!"
```
#
```
from itertools import product

def multi_dim_sum(dimensions):
    # dimensions is a list like [N1, N2, ..., Nm]
    
    # Generate all possible combinations of indices using itertools.product
    all_combinations = product(*(range(N + 1) for N in dimensions))
    
    total_sum = 0
    
    # Iterate over each combination and sum the values
    for combination in all_combinations:
        total_sum += sum(combination)
    
    return total_sum

# Example usage:
N1 = 3
N2 = 4
N3 = 2
print(multi_dim_sum([N1, N2]))    # 2D summation
print(multi_dim_sum([N1, N2, N3])) # 3D summation
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j], arr[i] = arr[i], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return arr, elapsed_time

def measure_time(N):
    arr = np.random.rand(N) * 100
    # Sort the array and measure time taken
    _, elapsed_time = bubble_sort(arr)
    return elapsed_time

# Array sizes from 2^0 to 2^12
array_sizes = [2**i for i in range(13)]
times = []

# Time taken for each array size
for N in array_sizes:
    t = measure_time(N)
    times.append(t)
    print(f"Time taken to sort array of size {N}: {t:.5f} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, times, marker='o', linestyle='--', color='b')
plt.xlabel('Array Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Bubble Sort Time Complexity')
plt.grid(True)
plt.show()
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j
```
#
```
#!/bin/bash

# Ensure the script is executable
chmod +x make_inp.py

# Read the number of molecules from indices.txt
Nmols=$(wc -l < indices.txt)

for imol in $(seq $Nmols); do
  # Get the folder name from folders.txt
  folder=$(sed -n "${imol}p" folders.txt)
  
  # Extract the current index from indices.txt
  sed -n "${imol}p" indices.txt > indscr.txt
  
  # Create the folder if it doesn't exist
  mkdir -p "../$folder"
  
  # Copy the template file and run the Python script
  cp 1AP_c2v.com inp_scr.com
  python3 make_inp.py

  # Move the generated opt.com to the new folder
  mv opt.com "../$folder"

  echo "Processed folder: $folder"
done

echo "All folders processed!"
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    start_time = time.time()
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Define the size range as powers of 2 from 2^0 to 2^13
sizes = [2**i for i in range(14)]

# Initialize lists to store times
times_bubble = []
times_quicksort = []
times_mergesort = []
times_heapsort = []

for size in sizes:
    array = np.random.rand(size)
    
    # Bubble Sort
    times_bubble.append(bubble_sort(array.copy()))
    
    # Quick Sort
    start_time = time.time()
    np.sort(array.copy(), kind='quicksort')
    end_time = time.time()
    times_quicksort.append(end_time - start_time)
    
    # Merge Sort
    start_time = time.time()
    np.sort(array.copy(), kind='mergesort')
    end_time = time.time()
    times_mergesort.append(end_time - start_time)
    
    # Heap Sort
    start_time = time.time()
    np.sort(array.copy(), kind='heapsort')
    end_time = time.time()
    times_heapsort.append(end_time - start_time)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the CPU times
plt.plot(sizes, times_bubble, label='Bubble Sort', marker='o')
plt.plot(sizes, times_quicksort, label='Quick Sort', marker='o')
plt.plot(sizes, times_mergesort, label='Merge Sort', marker='o')
plt.plot(sizes, times_heapsort, label='Heap Sort', marker='o')

# Plot aN^2 and bNlog(N)
a = 1e-7  # Adjustable constant
b = 1e-6  # Adjustable constant
plt.plot(sizes, [a*(n**2) for n in sizes], label='$aN^2$', linestyle='--')
plt.plot(sizes, [b*n*np.log(n) for n in sizes], label='$bN\log(N)$', linestyle='--')

# Labeling the plot
plt.xlabel('Array Size (N)')
plt.ylabel('CPU Time (s)')
plt.title('CPU Time for Various Sorting Algorithms')
plt.legend()
plt.grid(True)
plt.xscale('log', base=2)
plt.show()
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    start_time = time.time()
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Define the size range
sizes = range(20, 213)

# Initialize lists to store times
times_bubble = []
times_quicksort = []
times_mergesort = []
times_heapsort = []

for size in sizes:
    array = np.random.rand(size)
    
    # Bubble Sort
    times_bubble.append(bubble_sort(array.copy()))
    
    # Quick Sort
    start_time = time.time()
    np.sort(array.copy(), kind='quicksort')
    end_time = time.time()
    times_quicksort.append(end_time - start_time)
    
    # Merge Sort
    start_time = time.time()
    np.sort(array.copy(), kind='mergesort')
    end_time = time.time()
    times_mergesort.append(end_time - start_time)
    
    # Heap Sort
    start_time = time.time()
    np.sort(array.copy(), kind='heapsort')
    end_time = time.time()
    times_heapsort.append(end_time - start_time)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the CPU times
plt.plot(sizes, times_bubble, label='Bubble Sort', marker='o')
plt.plot(sizes, times_quicksort, label='Quick Sort', marker='o')
plt.plot(sizes, times_mergesort, label='Merge Sort', marker='o')
plt.plot(sizes, times_heapsort, label='Heap Sort', marker='o')

# Plot aN^2 and bNlog(N)
a = 1e-7  # Adjustable constant
b = 1e-6  # Adjustable constant
plt.plot(sizes, [a*(n**2) for n in sizes], label='$aN^2$', linestyle='--')
plt.plot(sizes, [b*n*np.log(n) for n in sizes], label='$bN\log(N)$', linestyle='--')

# Labeling the plot
plt.xlabel('Array Size (N)')
plt.ylabel('CPU Time (s)')
plt.title('CPU Time for Various Sorting Algorithms')
plt.legend()
plt.grid(True)
plt.show()
```
#
```for f in tddft/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDDFT/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
echo "--------------"
for f in tda/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDA/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
```
#
```
import csv

# Define the input and output CSV file paths
input_file = 'input.csv'
output_file = 'output.csv'

# Open the input CSV file for reading
with open(input_file, 'r') as infile:
    reader = csv.reader(infile)
    
    # Open the output CSV file for writing
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Skip the header in the input file
        header = next(reader)
        
        # Write the header for the new column
        writer.writerow(['New Column'])
        
        # Process each row
        for row in reader:
            # Check the value of the 3rd column (index 2)
            value = float(row[2])
            if value < 0.8:
                new_value = 1
            else:
                new_value = 0
            
            # Write the new value as a single-column row in the output file
            writer.writerow([new_value])
```
# 
```
import os

Nmol = 2285
geomfile = '33059_bnpah_TPSSh_def2SVP.xyz'

filedir = os.getcwd()

geom_file = open(geomfile, 'r')

for imol in range(Nmol):
    line = geom_file.readline().strip()
    
    if line:
        Nat = int(line)
        title = geom_file.readline().strip()

        mol = "Mol_{:05d}".format(imol+1)
        print(mol)

        # Create directory for each molecule
        mol_dir = os.path.join(filedir, mol)
        os.mkdir(mol_dir)

        # Read coordinates and write the sp.com file
        sp_file = os.path.join(mol_dir, 'sp.com')
        with open(sp_file, 'w') as inputfile:
            # Write Gaussian header
            inputfile.write('%mem=8GB\n')
            inputfile.write('%nprocs=2\n')
            inputfile.write('#RHF/cc-pvdz stable\n\n')
            inputfile.write(f'{mol}\n\n')
            inputfile.write('0 1\n')  # Charge and multiplicity
            
            # Write coordinates, skipping the first 2 lines
            for iat in range(1, Nat + 1):
                line = geom_file.readline().split()
                sym = line[0]
                R = [float(line[1]), float(line[2]), float(line[3])]
                inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

            # Gaussian footer
            inputfile.write('\n\n')

geom_file.close()
```
#
```import subprocess

def convert_gbw_to_molden(gbw_file):
    # Construct the command to convert the .gbw file to .molden.input
    command = f"/home/Lib/ORCA_600/orca_2mkl {gbw_file} -molden"
    
    try:
        # Run the command using subprocess
        process = subprocess.run(command, shell=True, check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Output the result of the conversion
        print("Conversion successful!")
        print("Output:\n", process.stdout)
        
    except subprocess.CalledProcessError as e:
        # Handle the error and print out what went wrong
        print(f"Error during conversion:\n{e.stderr}")

if __name__ == "__main__":
    # Replace 'tda' with the actual name of your .gbw file (without the .gbw extension)
    gbw_file = "tda"
    convert_gbw_to_molden(gbw_file)
```
#
```
import subprocess
import time

def run_multiwfn():
    # Open Multiwfn with the specified input file
    process = subprocess.Popen(['Multiwfn', 'tda.molden.input'], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
    
    # Define the sequence of inputs to Multiwfn
    inputs = [
        "18\n",                                  # Select option 18
        "14\n",                                  # Select option 14
        "/home/atreyee/project/azulene/NTO/tda.out\n",  # Provide the path to the file
        "3\n",                                   # Select option 3
        "1-6\n",                                 # Specify states 1-6
        "q\n"                                    # Exit
    ]
    
    # Send inputs to Multiwfn, with a slight delay between each
    for input_command in inputs:
        process.stdin.write(input_command)
        process.stdin.flush()
        time.sleep(0.5)  # Adjust the delay as necessary depending on your system's speed

    # Wait for the process to finish
    stdout, stderr = process.communicate()
    
    # Output the result for debugging
    if stdout:
        print("Multiwfn Output:\n", stdout)
    if stderr:
        print("Multiwfn Errors:\n", stderr)

if __name__ == "__main__":
    run_multiwfn()
```
#
```
import numpy as np

# Define the function f(a)
def f(a):
    return a**2 - 4  # Example function, modify as needed

# Bisection method parameters
thv = 1e-6  # Tolerance value
dx = 999  # Initial difference
a = 1  # Lower bound
b = 6  # Upper bound
xmid_old = b
steps = 0  # Step counter

# Bisection method loop
while dx > thv:
    xmid = (a + b) / 2  # Midpoint
    dx = np.abs(xmid - xmid_old)  # Update difference
    
    fa = f(a)
    fb = f(b)
    fmid = f(xmid)

    if fmid * fb > 0:
        b = xmid
    else:
        a = xmid
    
    xmid_old = xmid  # Update the old midpoint
    steps += 1  # Increment step counter

# Output the root found and the number of steps
print(f"Root is approximately: {xmid}")
print(f"Number of steps: {steps}")
```
#
```
import numpy as np

# Define the function f(a)
def f(a):
    return a**2 - 4  # Example function, modify as needed

# Bisection method parameters
thv = 1e-6  # Tolerance value
dx = 999  # Initial difference
a = 1  # Lower bound
b = 6  # Upper bound
xmid_old = b

# Bisection method loop
while dx > thv:
    xmid = (a + b) / 2  # Midpoint
    dx = np.abs(xmid - xmid_old)  # Update difference
    
    fa = f(a)
    fb = f(b)
    fmid = f(xmid)

    if fmid * fb > 0:
        b = xmid
    else:
        a = xmid
    
    xmid_old = xmid  # Update the old midpoint

# Output the root found
print(f"Root is approximately: {xmid}")
```
#
```
import numpy as np

# Define the function f(a)
def f(a):
    return a**2 - 4  # Example function, modify as needed

# Bisection method parameters
thv = 1e-6  # Tolerance value
dx = 999  # Initial difference
a = 1  # Lower bound
b = 6  # Upper bound
xmid_old = b

# Bisection method loop
while dx > thv:
    xmid = (a + b) / 2  # Midpoint
    dx = np.abs(xmid - xmid_old)  # Update difference
    
    fa = f(a)
    fb = f(b)
    fmid = f(xmid)

    if fmid * fb > 0:
        b = xmid
    else:
        a = xmid
    
    xmid_old = xmid  # Update the old midpoint

# Output the root found
print(f"Root is approximately: {xmid}")
```
#
```
from itertools import product

def multi_dim_sum(dimensions):
    # dimensions is a list like [N1, N2, ..., Nm]
    
    # Generate all possible combinations of indices using itertools.product
    all_combinations = product(*(range(N + 1) for N in dimensions))
    
    total_sum = 0
    
    # Iterate over each combination and sum the values
    for combination in all_combinations:
        total_sum += sum(combination)
    
    return total_sum

# Example usage:
N1 = 3
N2 = 4
N3 = 2
print(multi_dim_sum([N1, N2]))    # 2D summation
print(multi_dim_sum([N1, N2, N3])) # 3D summation
```
#
```
#!/bin/bash

# Ensure the script is executable
chmod +x make_inp.py

# Read the number of molecules from indices.txt
Nmols=$(wc -l < indices.txt)

for imol in $(seq $Nmols); do
  # Get the folder name from folders.txt
  folder=$(sed -n "${imol}p" folders.txt)
  
  # Extract the current index from indices.txt
  sed -n "${imol}p" indices.txt > indscr.txt
  
  # Create the folder if it doesn't exist
  mkdir -p "../$folder"
  
  # Copy the template file and run the Python script
  cp 1AP_c2v.com inp_scr.com
  python3 make_inp.py

  # Move the generated opt.com to the new folder
  mv opt.com "../$folder"

  echo "Processed folder: $folder"
done

echo "All folders processed!"
```
#
```
#!/bin/bash

# Get the number of molecules
Nmols=$(wc -l < indices.txt)

# Loop over each molecule index
for imol in $(seq 1 $Nmols); do

  # Extract the folder name and indices for the current molecule
  folder=$(sed -n "${imol}p" folders.txt)
  sed -n "${imol}p" indices.txt > indscr.txt
  
  # Create the directory if it doesn't exist
  mkdir -p "../$folder"

  echo "Processing folder: $folder"

  # Copy the template file to the folder
  cp 1AP_c2v.com "$folder/1AP_c2v.com"

  # Move to the directory
  cd "$folder"

  # Run the Python script to generate the input file
  python3 ../make_inp.py

  # Move the generated opt.com file to the parent directory
  mv opt.com "../$folder/${folder}_opt.com"

  # Go back to the parent directory
  cd ..

  echo "Processed folder: $folder"

done

echo "All folders processed!"
```
#
```
import numpy as np

# Define the function f(a)
def f(a):
    return a**2 - 4  # Example function, modify as needed

# Bisection method parameters
thv = 1e-6  # Tolerance value
dx = 999  # Initial difference
a = 1  # Lower bound
b = 6  # Upper bound
xmid_old = b

# Bisection method loop
while dx > thv:
    xmid = (a + b) / 2  # Midpoint
    dx = np.abs(xmid - xmid_old)  # Update difference
    
    fa = f(a)
    fb = f(b)
    fmid = f(xmid)

    if fmid * fb > 0:
        b = xmid
    else:
        a = xmid
    
    xmid_old = xmid  # Update the old midpoint

# Output the root found
print(f"Root is approximately: {xmid}")

``
#
```
import os

Nmol = 2285
geomfile = '33059_bnpah_TPSSh_def2SVP.xyz'

filedir = os.getcwd()

geom_file = open(geomfile, 'r')

for imol in range(Nmol):
    line = geom_file.readline().strip()
    
    if line:
        Nat = int(line)
        title = geom_file.readline().strip()

        mol = "Mol_{:05d}".format(imol+1)
        print(mol)

        # Create directory for each molecule
        mol_dir = os.path.join(filedir, mol)
        os.mkdir(mol_dir)

        # Read coordinates and write the sp.com file
        sp_file = os.path.join(mol_dir, 'sp.com')
        with open(sp_file, 'w') as inputfile:
            # Write Gaussian header
            inputfile.write('%mem=8GB\n')
            inputfile.write('%nprocs=2\n')
            inputfile.write('#RHF/cc-pvdz stable\n\n')
            inputfile.write(f'{mol}\n\n')
            inputfile.write('0 1\n')  # Charge and multiplicity
            
            # Write coordinates, skipping the first 2 lines
            for iat in range(1, Nat + 1):
                line = geom_file.readline().split()
                sym = line[0]
                R = [float(line[1]), float(line[2]), float(line[3])]
                inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

            # Gaussian footer
            inputfile.write('\n\n')

geom_file.close()
```
#
```
CCSD_VDZ_dir=CCSD_VDZ
CCSD_VTZ_dir=CCSD_VTZ
CCSDT_VDZ_dir=CCSDT_VDZ

dist_file=dist_1AP_D3h.csv
angle_file=angle_1AP_D3h.csv

# Get the number of coordinates from the CCSDT_VDZ folder
Ncoord=$(grep 'Number of displacements for' ../${CCSDT_VDZ_dir}/subfolder/opt.log | awk '{print $7/2}' | head -1)

# Extract optimized variables from the respective opt.out files
grep -$Ncoord ' Optimized variables' ../$CCSD_VDZ_dir/subfolder/opt.out | tail -$Ncoord > CCSD_VDZ.txt
grep -$Ncoord ' Optimized variables' ../$CCSD_VTZ_dir/subfolder/opt.out | tail -$Ncoord > CCSD_VTZ.txt
grep -$Ncoord ' Optimized variables' ../$CCSDT_VDZ_dir/subfolder/opt.out | tail -$Ncoord > CCSDT_VDZ.txt

# Create distance and angle files
paste -d ' ' CCSDT_VDZ.txt CCSD_VTZ.txt CCSD_VDZ.txt | column -t | grep ANG | awk '{print $2+$5-$8","$11}' > $dist_file
paste -d ' ' CCSDT_VDZ.txt CCSD_VTZ.txt CCSD_VDZ.txt | column -t | grep DEGREE | awk '{print $2+$5-$8","$11}' > $angle_file

# Create test.com from opt.com inside the subfolder
cp ../$CCSDT_VDZ_dir/subfolder/opt.com test.com

# Add the necessary data to test.com
paste -d ' ' CCSDT_VDZ.txt CCSD_VTZ.txt CCSD_VDZ.txt | column -t | awk '{print $1,$2+$5-$8,$3}' | column -t >> test.com

# Add additional commands to the input file
echo "" >> test.com
echo "basis=STO-3G" >> test.com
echo "hf"  >> test.com
echo "put,XYZ,test.xyz" >> test.com
```
#
```
for f in tddft/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDDFT/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
echo "--------------"
for f in tda/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDA/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
```
#
folders

dist_file=dist.csv
angle_file=angle.csv

# Loop over folders starting with 1 in CCSD_VDZ, CCSD_VTZ, and CCSDT_VDZ directories
for folder in ../${CCSDT_VDZ_dir}/1*; do
  mol=$(basename "$folder")

  # Create corresponding folder directly in the existing extrapolate directory
  mkdir -p ${mol}

  # Get the number of coordinates from the CCSDT_VDZ folder
  Ncoord=$(grep 'Number of displacements for' ../${CCSDT_VDZ_dir}/${mol}/opt.log | head -1 | awk '{print $7/2}')

  # Extract optimized variables from the respective opt.out files
  grep -$Ncoord ' Optimized variables' ../$CCSD_VDZ_dir/${mol}/opt.out | tail -$Ncoord > ${mol}/CCSD_VDZ.txt
  grep -$Ncoord ' Optimized variables' ../$CCSD_VTZ_dir/${mol}/opt.out | tail -$Ncoord > ${mol}/CCSD_VTZ.txt
  grep -$Ncoord ' Optimized variables' ../$CCSDT_VDZ_dir/${mol}/opt.out | tail -$Ncoord > ${mol}/CCSDT_VDZ.txt

  # Create distance and angle files
  paste -d ' ' ${mol}/CCSDT_VDZ.txt ${mol}/CCSD_VTZ.txt ${mol}/CCSD_VDZ.txt | column -t | grep ANG | awk '{print $2+$5-$8","$11}' > ${mol}/$dist_file
  paste -d ' ' ${mol}/CCSDT_VDZ.txt ${mol}/CCSD_VTZ.txt ${mol}/CCSD_VDZ.txt | column -t | grep DEGREE | awk '{print $2+$5-$8","$11}' > ${mol}/$angle_file

  # Create test.com from opt.com inside the subfolder
  cp ../$CCSDT_VDZ_dir/${mol}/opt.com ${mol}/test.com

  # Modify test.com
  sed -i '/ANG/d' ${mol}/test.com
  sed -i '/DEGREE/d' ${mol}/test.com
  sed -i '/basis/d' ${mol}/test.com
  sed -i '/hf/d' ${mol}/test.com
  sed -i '/ccsd/d' ${mol}/test.com
  sed -i '/opt/d' ${mol}/test.com

  # Add necessary data to test.com
  paste -d ' ' ${mol}/CCSDT_VDZ.txt ${mol}/CCSD_VTZ.txt ${mol}/CCSD_VDZ.txt | column -t | awk '{print $1,$2+$5-$8,$3}' | column -t >> ${mol}/test.com

  # Add additional commands to the input file
  echo "" >> ${mol}/test.com
  echo "basis=STO-3G" >> ${mol}/test.com
  echo "hf" >> ${mol}/test.com
  echo "put,XYZ,test.xyz" >> ${mol}/test.com
done

```
#
```import os
import shutil

# Define the paths
extrapolate_folder = './extrapolate'
adc2_folder = './adc2'

# Create the adc2 folder if it doesn't exist
if not os.path.exists(adc2_folder):
    os.makedirs(adc2_folder)

# Loop through each folder inside extrapolate
for folder_name in os.listdir(extrapolate_folder):
    folder_path = os.path.join(extrapolate_folder, folder_name)
    
    # Check if it is a directory
    if os.path.isdir(folder_path):
        xyz_file = os.path.join(folder_path, 'test.xyz')
        
        # Read the test.xyz file, skipping the first two lines
        with open(xyz_file, 'r') as xyz:
            lines = xyz.readlines()[2:]
        
        # Prepare the Q-Chem input template
        qchem_input = '''$molecule
  0  1

'''
        # Add coordinates
        qchem_input += ''.join(lines)
        qchem_input += '''$end

$rem
jobtype             sp
method              adc(2)
basis               cc-pVTZ
aux_basis           rimp2-cc-pVTZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
'''
        
        # Create the corresponding folder inside adc2
        new_folder = os.path.join(adc2_folder, folder_name)
        os.makedirs(new_folder, exist_ok=True)
        
        # Write the all.com file inside the new folder
        allcom_file = os.path.join(new_folder, 'all.com')
        with open(allcom_file, 'w') as allcom:
            allcom.write(qchem_input)

print("Files created successfully!")
```
#
```
for f in tddft/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDDFT/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
echo "--------------"
for f in tda/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDA/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
```
#
```
import os

Nmol = 2285
geomfile = '33059_bnpah_TPSSh_def2SVP.xyz'

filedir = os.getcwd()

geom_file = open(geomfile, 'r')

for imol in range(Nmol):
    line = geom_file.readline().strip()
    
    if line:
        Nat = int(line)
        title = geom_file.readline().strip()

        mol = "Mol_{:05d}".format(imol+1)
        print(mol)

        # Create directory for each molecule
        mol_dir = os.path.join(filedir, mol)
        os.mkdir(mol_dir)

        # Read coordinates and write the sp.com file
        sp_file = os.path.join(mol_dir, 'sp.com')
        with open(sp_file, 'w') as inputfile:
            # Write Gaussian header
            inputfile.write('%mem=8GB\n')
            inputfile.write('%nprocs=2\n')
            inputfile.write('#RHF/cc-pvdz stable\n\n')
            inputfile.write(f'{mol}\n\n')
            inputfile.write('0 1\n')  # Charge and multiplicity
            
            # Write coordinates, skipping the first 2 lines
            for iat in range(1, Nat + 1):
                line = geom_file.readline().split()
                sym = line[0]
                R = [float(line[1]), float(line[2]), float(line[3])]
                inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

            # Gaussian footer
            inputfile.write('\n\n')

geom_file.close()
```
#
```
import os
import shutil

# Define the paths
extrapolate_folder = './extrapolate'
adc2_folder = './adc2'

# Create the adc2 folder if it doesn't exist
if not os.path.exists(adc2_folder):
    os.makedirs(adc2_folder)

# Loop through each folder inside extrapolate
for folder_name in os.listdir(extrapolate_folder):
    folder_path = os.path.join(extrapolate_folder, folder_name)
    
    # Check if it is a directory
    if os.path.isdir(folder_path):
        xyz_file = os.path.join(folder_path, 'test.xyz')
        
        # Read the test.xyz file, skipping the first two lines
        with open(xyz_file, 'r') as xyz:
            lines = xyz.readlines()[2:]
        
        # Prepare the Q-Chem input template
        qchem_input = '''$molecule
  0  1

'''
        # Add coordinates
        qchem_input += ''.join(lines)
        qchem_input += '''$end

$rem
jobtype             sp
method              adc(2)
basis               cc-pVTZ
aux_basis           rimp2-cc-pVTZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
'''
        
        # Create the corresponding folder inside adc2
        new_folder = os.path.join(adc2_folder, folder_name)
        os.makedirs(new_folder, exist_ok=True)
        
        # Write the all.com file inside the new folder
        allcom_file = os.path.join(new_folder, 'all.com')
        with open(allcom_file, 'w') as allcom:
            allcom.write(qchem_input)

print("Files created successfully!")

```
#
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Conversion factor from Hartree to kcal/mol
hartree2kcm = 627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_MP2.csv', header=None)

# Extract x and z values
x = data.iloc[:, 0]
z = data.iloc[:, 1]

# Convert energy values
z = z - np.min(z)  # Subtract the minimum value from z
z = z * hartree2kcm  # Convert to kcal/mol

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, z, label='Energy')
ax.set_title('Energy Plot')
ax.set_xlabel("Values")
ax.set_ylabel("Energy [$\\Delta E$ in kcal/mol]")
ax.legend()

# Save and show the plot
plt.savefig('energy_plot.png')
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the 2nd column (index 1) with respect to the 1st column (index 0)
plt.plot(df[0], df[1])
plt.title('Plot of the 2nd Column vs 1st Column')
plt.xlabel('1st Column')
plt.ylabel('2nd Column')
plt.grid(True)
plt.show()
```
```
import os
import shutil

# Define the paths
extrapolate_folder = './extrapolate'
adc2_folder = './adc2'

# Create the adc2 folder if it doesn't exist
if not os.path.exists(adc2_folder):
    os.makedirs(adc2_folder)

# Loop through each folder inside extrapolate
for folder_name in os.listdir(extrapolate_folder):
    folder_path = os.path.join(extrapolate_folder, folder_name)
    
    # Check if it is a directory
    if os.path.isdir(folder_path):
        xyz_file = os.path.join(folder_path, 'test.xyz')
        
        # Read the test.xyz file, skipping the first two lines
        with open(xyz_file, 'r') as xyz:
            lines = xyz.readlines()[2:]
        
        # Prepare the Q-Chem input template
        qchem_input = '''$molecule
  0  1

'''
        # Add coordinates
        qchem_input += ''.join(lines)
        qchem_input += '''$end

$rem
jobtype             sp
method              adc(2)
basis               cc-pVTZ
aux_basis           rimp2-cc-pVTZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
'''
        
        # Create the corresponding folder inside adc2
        new_folder = os.path.join(adc2_folder, folder_name)
        os.makedirs(new_folder, exist_ok=True)
        
        # Write the all.com file inside the new folder
        allcom_file = os.path.join(new_folder, 'all.com')
        with open(allcom_file, 'w') as allcom:
            allcom.write(qchem_input)

print("Files created successfully!")
```
#
```for f in tddft/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDDFT/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
echo "--------------"
for f in tda/*/tddft.log; do
   T1=$(grep 'Excited State '  $f | grep Triplet | head -1 | awk '{printf "%7.4f\n", $5}' )
   T2=$(grep 'Excited State '  $f | grep Triplet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S1=$(grep 'Excited State '  $f | grep Singlet | head -1 | awk '{printf "%7.4f\n", $5}' )
   S2=$(grep 'Excited State '  $f | grep Singlet | head -2 | tail -1 | awk '{printf "%7.4f\n", $5}' )
   S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
   echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2
done

f=SCS-PBE-QIDH/TDA/tddft.out.bz2
S1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -1 | awk '{printf "%7.4f\n", $2}' )
S2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -2 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T1=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -7 | tail -1 | awk '{printf "%7.4f\n", $2}' )
T2=$(bzgrep -A16  '        ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' $f  | tail -12 | head -8 | tail -1 | awk '{printf "%7.4f\n", $2}' )

S2_2T1=$( echo $S2 $T1 | awk '{printf "%7.4f\n", $1-2*$2 }' )
S2_2T2=$( echo $S2 $T2 | awk '{printf "%7.4f\n", $1-2*$2 }' )
echo $f $S1 $S2 $T1 $T2 $S2_2T1 $S2_2T2

```
#
```# Define the data for sets a, b, c, d
a = {"Set": "a", "R1": 0.00042267, "R2": 0.00027884, "A11": 0.00470327, "R3": 0.00111791, "R41": 0.00022695, "A21": -0.01103988}
b = {"Set": "b", "R1": -0.00081553, "R2": -0.0011027, "A11": 0.00124591, "R3": -0.00094836, "R41": -0.00093005, "A21": 0.00786545}
c = {"Set": "c", "R1": -0.00496753, "R2": -0.00557967, "A11": 0.04678181, "R3": -0.01093188, "R41": -0.00427303, "A21": 0.03141856}
d = {"Set": "d", "R1": -0.00516753, "R2": -0.00527967, "A11": 0.04528181, "R3": -0.00953188, "R41": -0.00357303, "A21": -0.01168144}

# Define parameters and sets
parameters = ["R1 (ANG)", "R2 (ANG)", "A11 (DEGREE)", "R3 (ANG)", "R41 (ANG)", "A21 (DEGREE)"]
sets = [a, b, c, d]

# Print the values in separate columns
for param in parameters:
    print("{:<12} {:<12.8f} {:<12.8f} {:<12.8f} {:<12.8f}".format(
        param, a[param.split()[0]], b[param.split()[0]], c[param.split()[0]], d[param.split()[0]]))
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files without headers
csv1 = 'tda_results.csv'
csv2 = 'tddft_results.csv'

df1 = pd.read_csv(csv1, header=None)
df2 = pd.read_csv(csv2, header=None)

# Filter the rows based on the given ranges
filtered_df1 = df1[(df1.iloc[:, 8] >= 1.5) & (df1.iloc[:, 8] <= 4.0)]
filtered_df2 = df2[(df2.iloc[:, 8] >= 0.0) & (df2.iloc[:, 8] <= 2.0)]

# Find common indices in both filtered DataFrames
common_indices = filtered_df1.index.intersection(filtered_df2.index)

# Filter the DataFrames again to only include common indices
filtered_df1 = filtered_df1.loc[common_indices]
filtered_df2 = filtered_df2.loc[common_indices]

# Plot a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(filtered_df1.iloc[:, 8], filtered_df2.iloc[:, 8], c='blue', label='Filtered Data')
plt.xlabel('tda')
plt.ylabel('tddft')
plt.title('Scatter Plot')
plt.legend()
plt.grid(True)
plt.axis('square')
plt.show()

# Prepare the data for output
tda_column1 = filtered_df1.iloc[:, 0].reset_index(drop=True)
tda_column2 = filtered_df1.iloc[:, 1].reset_index(drop=True)
tda_column8 = filtered_df1.iloc[:, 8].astype(str).reset_index(drop=True)
tddft_column8 = filtered_df2.iloc[:, 8].astype(str).reset_index(drop=True)

# Combine columns with underscore
combined_column8 = tda_column8 + '_' + tddft_column8

# Create DataFrame for output
combined_data = pd.DataFrame({
    'Column1': tda_column1,
    'Column2': tda_column2,
    'CombinedColumn8': combined_column8
})

# Save to .smi file
combined_data.to_csv('combined_results.smi', sep='\t', index=False, header=False)

print("Data has been saved to 'combined_results.smi'.")
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'scan_all_data.csv'
try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Conversion factor from Hartree to eV
conversion_factor = 27.2114

# Define the specific values for the first column
x_values = [-2.5, -2, -1.8, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 1.8, 2, 2.5]

# Filter rows where the first column has the specified values and the 3rd, 4th, and 5th columns have non-missing values
filtered_df = df[df[df.columns[0]].isin(x_values)].dropna(subset=[df.columns[1], df.columns[2], df.columns[3]])

# Extract the relevant columns
x = filtered_df.iloc[:, 0]
y1 = filtered_df.iloc[:, 1]
y2 = filtered_df.iloc[:, 2]
y3 = filtered_df.iloc[:, 3]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each y column
plt.plot(x, y1 * conversion_factor, label='S$_1$', marker='o')
plt.plot(x, y2 * conversion_factor, label='T$_1$', marker='o')
plt.plot(x, y3 * conversion_factor, label='STG', marker='o')

# Setting the x-axis limits
plt.xlim(-2.5, 2.5)
# plt.ylim(-1, 2.0)

# Adding labels and title
plt.xlabel('DNC')
plt.ylabel('Energy (eV)')
plt.title('Scan plot of the excited states and STG')
plt.legend()

# Show plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'scan_all_data.csv'
try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Conversion factor from Hartree to eV
conversion_factor = 27.2114

# Define the specific values for the first column
x_values = [-2.5, -2, -1.8, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 1.8, 2, 2.5]

# Filter rows where the first column has the specified values and the 3rd, 4th, and 5th columns have non-missing values
filtered_df = df[df[df.columns[0]].isin(x_values)].dropna(subset=[df.columns[2], df.columns[3], df.columns[4]])

# Extract the relevant columns
x = filtered_df.iloc[:, 0]
y1 = filtered_df.iloc[:, 2]
y2 = filtered_df.iloc[:, 3]
y3 = filtered_df.iloc[:, 4]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each y column
plt.plot(x, y1 * conversion_factor, label='S$_1$', marker='o')
plt.plot(x, y2 * conversion_factor, label='T$_1$', marker='o')
plt.plot(x, y3 * conversion_factor, label='STG', marker='o')

# Setting the x-axis limits
plt.xlim(-2.5, 2.5)
# plt.ylim(-1, 2.0)

# Adding labels and title
plt.xlabel('DNC')
plt.ylabel('Energy (eV)')
plt.title('Scan plot of the excited states and STG')
plt.legend()

# Show plot
plt.show()
```
#
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    start_time = time.time()
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                arr[j
```
#
```
import os

def create_folders_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    # Open and read the file
    with open(file_path, 'r') as file:
        folder_names = file.readlines()

    # Create folders for each name in the file
    for folder_name in folder_names:
        folder_name = folder_name.strip()  # Remove any leading/trailing whitespace
        if folder_name:  # Ensure the folder name is not empty
            sanitized_folder_name = folder_name.replace(',', '_').replace('-', '_')
            try:
                os.makedirs(sanitized_folder_name, exist_ok=True)
                print(f"Created folder: {sanitized_folder_name}")
            except OSError as e:
                print(f"Error creating folder {sanitized_folder_name}: {e}")

# Specify the path to your text file
file_path = 'a.txt'

# Call the function to create folders
create_folders_from_file(file_path)
```
#
```
from tabulate import tabulate

# Define the data for sets a, b, c, d
a = {"Set": "a", "R1": 0.00042267, "R2": 0.00027884, "A11": 0.00470327, "R3": 0.00111791, "R41": 0.00022695, "A21": -0.01103988}
b = {"Set": "b", "R1": -0.00081553, "R2": -0.0011027, "A11": 0.00124591, "R3": -0.00094836, "R41": -0.00093005, "A21": 0.00786545}
c = {"Set": "c", "R1": -0.00496753, "R2": -0.00557967, "A11": 0.04678181, "R3": -0.01093188, "R41": -0.00427303, "A21": 0.03141856}
d = {"Set": "d", "R1": -0.00516753, "R2": -0.00527967, "A11": 0.04528181, "R3": -0.00953188, "R41": -0.00357303, "A21": -0.01168144}

# Extract headers and data
headers = ["Parameter", "a", "b", "c", "d"]
data = [
    ["R1 (ANG)", a["R1"], b["R1"], c["R1"], d["R1"]],
    ["R2 (ANG)", a["R2"], b["R2"], c["R2"], d["R2"]],
    ["A11 (DEGREE)", a["A11"], b["A11"], c["A11"], d["A11"]],
    ["R3 (ANG)", a["R3"], b["R3"], c["R3"], d["R3"]],
    ["R41 (ANG)", a["R41"], b["R41"], c["R41"], d["R41"]],
    ["A21 (DEGREE)", a["A21"], b["A21"], c["A21"], d["A21"]]
]

# Print the table
print(tabulate(data, headers=headers, tablefmt="grid"))
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Define the energies and custom labels
data = {
    'Label': ['a', '2', '3', '4', '6', '8'],
    'Energy': [
        -385.691580683,
        -770.860688368 / 2,
        -771.336917778 / 2,
        -771.380824354 / 2,
        -771.382283738 / 2,
        -771.382816488 / 2
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the line passing through the first point
plt.figure(figsize=(10, 6))
plt.plot([1, len(df)], [df['Energy'][0], df['Energy'][0]], color='r', label='Line through Point a')

# Plot the curve joining the other points
plt.plot(range(2, len(df) + 1), df['Energy'][1:], marker='o', linestyle='-', color='b', label='Curve through other points')

# Add labels to each point
for i, row in df.iterrows():
    plt.text(i + 1, row['Energy'], f'{row["Label"]} ({row["Energy"]:.6f})', fontsize=10, ha='right')

# Add title and labels
plt.title('Energy Plot')
plt.xlabel('Index')
plt.ylabel('Energy')
plt.legend()

# Save the figure as a PDF
plt.savefig('energies_plot.pdf')

# Show the plot
plt.show()
```
#
```
import matplotlib.pyplot as plt

# Energies to plot
energies = [
    -385.691580683,
    -770.860688368 / 2,
    -771.336917778 / 2,
    -771.380824354 / 2,
    -771.382283738 / 2,
    -771.382816488 / 2
]

# Labels for each point
labels = [
    'Point 1',
    'Point 2',
    'Point 3',
    'Point 4',
    'Point 5',
    'Point 6'
]

# X-axis values (indices)
x_values = list(range(1, len(energies) + 1))

# Plot the energies
plt.figure(figsize=(10, 6))
plt.plot(x_values, energies, marker='o', linestyle='-', color='b')

# Add labels to each point
for i, (x, y) in enumerate(zip(x_values, energies)):
    plt.text(x, y, f'{labels[i]} ({y:.6f})', fontsize=10, ha='right')

# Add title and labels
plt.title('Energy Plot')
plt.xlabel('Index')
plt.ylabel('Energy')

# Save the figure as a PDF
plt.savefig('energies_plot.pdf')

# Show the plot
plt.show()
```
#
```
import pandas as pd

# Load the CSV files
csv_16_columns = pd.read_csv('csv_16_columns.csv', header=None)
csv_1_column = pd.read_csv('csv_1_column.csv', header=None)

# Ensure the single column DataFrame has the same number of rows as the 16 columns DataFrame
if len(csv_16_columns) != len(csv_1_column):
    raise ValueError("The number of rows in the two Cimport pandas as pd

# Load the CSV files
csv_16_columns = pd.read_csv('csv_16_columns.csv', header=None)
csv_1_column = pd.read_csv('csv_1_column.csv', header=None)

# Ensure the single column DataFrame has the same number of rows as the 16 columns DataFrame
if len(csv_16_columns) != len(csv_1_column):
    raise ValueError("The number of rows in the two CSV files does not match.")

# Add the single column from the second CSV to the first CSV
csv_merged = pd.concat([csv_16_columns, csv_1_column], axis=1)

# Save the merged DataFrame to a new CSV file
csv_merged.to_csv('csv_merged.csv', index=False, header=False)
SV files does not match.")

# Add the single column from the second CSV to the first CSV
csv_merged = pd.concat([csv_16_columns, csv_1_column], axis=1)

# Save the merged DataFrame to a new CSV file
csv_merged.to_csv('csv_merged.csv', index=False, header=False)
```
#
```
import glob

# Define the folder where the CSV files are located
folder_path = './'  # Adjust this path if needed

# Get all CSV files in the folder
csv_files = glob.glob(f'{folder_path}/*.csv')

# Prepare a list to hold the formatted rows
formatted_rows = []

# Process each CSV file
for file in csv_files:
    # Read the CSV file without headers
    with open(file, 'r') as f:
        # Skip the first line (header)
        next(f)
        # Read each line and append to the list
        for line in f:
            # Remove any leading/trailing whitespace and split by whitespace
            row = line.strip().split()
            # Join the row elements with commas and add to formatted_rows
            formatted_rows.append(row)  # Store as lists for sorting

# Sort rows based on the first column (convert to float for sorting)
formatted_rows.sort(key=lambda x: float(x[0]))

# Write the sorted rows to a new CSV file with commas
with open('all.csv', 'w') as f:
    for row in formatted_rows:
        f.write(','.join(map(str, row)) + '\n')

# Calculate the maximum width of each column for alignment
max_lengths = [max(len(str(item)) for item in col) for col in zip(*formatted_rows)]

# Create a list to hold the aligned output
aligned_output = []
for row in formatted_rows:
    # Format each row with proper spacing based on max_lengths
    aligned_row = '   '.join(f'{str(item):<{max_lengths[i]}}' for i, item in enumerate(row))
    aligned_output.append(aligned_row)

# Write the aligned output to a text file
with open('aligned_output.txt', 'w') as f:
    for row in aligned_output:
        f.write(row + '\n')

print("CSV files have been merged, sorted, and written to 'all.csv'.")
print("An aligned output has been written to 'aligned_output.txt'.")
```
#
```
import matplotlib.pyplot as plt

# Energies to plot
energies = [
    -385.691580683,
    -770.860688368 / 2,
    -771.336917778 / 2,
    -771.380824354 / 2,
    -771.382283738 / 2,
    -771.382816488 / 2
]

# Labels for each point
labels = [
    'Point 1',
    'Point 2',
    'Point 3',
    'Point 4',
    'Point 5',
    'Point 6'
]

# X-axis values (indices)
x_values = list(range(1, len(energies) + 1))

# Plot the energies
plt.figure(figsize=(10, 6))
plt.plot(x_values, energies, marker='o', linestyle='-', color='b')

# Add labels to each point
for i, (x, y) in enumerate(zip(x_values, energies)):
    plt.text(x, y, f'{labels[i]} ({y:.6f})', fontsize=10, ha='right')

# Add title and labels
plt.title('Energy Plot')
plt.xlabel('Index')
plt.ylabel('Energy')

# Save the figure as a PDF
plt.savefig('energies_plot.pdf')

# Show the plot
plt.show()
```
#
```
import os
import re

# Define the template folder and file
output_base_folder = '/home/atreyee/Project_AP1XY/all_aza/C3h_from_D3h/contour_plot_CCSDT_at_MP2'
template_folder = output_base_folder + '/template'
template_file = 'opt.com'
output_MP2 = '/home/atreyee/Project_AP1XY/all_aza/C3h_from_D3h/contour_plot_at_MP2'

# Define the ranges and increments for var1 and var2
var1_start = 1.3
var1_end = 1.5
var1_increment = 0.01
var2_start = 1.3
var2_end = 1.5
var2_increment = 0.01

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
        if float(var1) >= float(var2):
            # Create the directory name
            folder_name = f'Mol_{var1:.2f}_{var2:.2f}'
            folder_path = os.path.join(output_base_folder, folder_name)

            # Create the directory if it does not exist
            os.makedirs(folder_path, exist_ok=True)

            # Read the template file
            template_subfolder = template_folder + ('_diag' if var1 == var2 else '_offdiag')
            with open(os.path.join(template_subfolder, template_file), 'r') as file:
                template_content = file.read()

            # Replace variables with actual values
            modified_content = template_content.replace('var1', str(var1)).replace('var2', str(var2))
            
            # Write the modified content to a new opt.com file in the new directory
            new_file_path = os.path.join(folder_path, template_file)
            with open(new_file_path, 'w') as new_file:
                new_file.write(modified_content)

            # Read and format optimized variables from opt.out
            opt_out_path = os.path.join(output_MP2, folder_name, "opt.out")
            with open(opt_out_path, 'r') as opt_out_file:
                lines = opt_out_file.readlines()

            # Extract optimized variables with controlled precision
            optimized_vars = []
            capture = False
            for line in lines:
                if 'Optimized variables' in line:
                    capture = True
                    continue
                if capture:
                    if re.search(r'^\s*\*', line):  # Stops at the star line if found
                        break
                    # Format each variable to 6 decimal places to avoid overflow stars
                    formatted_line = re.sub(r'(\d+\.\d+)', lambda m: f"{float(m.group(0)):.6f}", line)
                    optimized_vars.append(formatted_line.strip())

            # Append formatted optimized variables and method details
            with open(new_file_path, 'a') as new_file:
                new_file.write('\n'.join(optimized_vars) + '\n')
                new_file.write("\n\nbasis=cc-pVDZ\nhf\nccsd(t)\n")

print("All combinations have been created.")
```
#
```
import os
import re

# Define the template folder and file
output_base_folder = '/home/atreyee/Project_AP1XY/all_aza/C3h_from_D3h/contour_plot_CCSDT_at_MP2'
template_folder = output_base_folder + '/template'
template_file = 'opt.com'
output_MP2 = '/home/atreyee/Project_AP1XY/all_aza/C3h_from_D3h/contour_plot_at_MP2'

# Define the ranges and increments for var1 and var2
var1_start = 1.3
var1_end = 1.5
var1_increment = 0.01
var2_start = 1.3
var2_end = 1.5
var2_increment = 0.01

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
        if float(var1) >= float(var2):
            # Create the directory name
            folder_name = f'Mol_{var1:.2f}_{var2:.2f}'
            folder_path = os.path.join(output_base_folder, folder_name)

            # Create the directory if it does not exist
            os.makedirs(folder_path, exist_ok=True)

            # Read the template file
            template_subfolder = template_folder + ('_diag' if var1 == var2 else '_offdiag')
            with open(os.path.join(template_subfolder, template_file), 'r') as file:
                template_content = file.read()

            # Replace variables with actual values
            modified_content = template_content.replace('var1', str(var1)).replace('var2', str(var2))
            
            # Write the modified content to a new opt.com file in the new directory
            new_file_path = os.path.join(folder_path, template_file)
            with open(new_file_path, 'w') as new_file:
                new_file.write(modified_content)

            # Read and format optimized variables from opt.out
            opt_out_path = os.path.join(output_MP2, folder_name, "opt.out")
            with open(opt_out_path, 'r') as opt_out_file:
                lines = opt_out_file.readlines()

            # Extract optimized variables with controlled precision
            optimized_vars = []
            capture = False
            for line in lines:
                if 'Optimized variables' in line:
                    capture = True
                    continue
                if capture:
                    if re.search(r'^\s*\*', line):  # Stops at the star line if found
                        break
                    # Format each variable to 6 decimal places to avoid overflow stars
                    formatted_line = re.sub(r'(\d+\.\d+)', lambda m: f"{float(m.group(0)):.6f}", line)
                    optimized_vars.append(formatted_line.strip())

            # Append formatted optimized variables and method details
            with open(new_file_path, 'a') as new_file:
                new_file.write('\n'.join(optimized_vars) + '\n')
                new_file.write("\n\nbasis=cc-pVDZ\nhf\nccsd(t)\n")

print("All combinations have been created.")
```
#
```
import os

# Define the template folder and file
output_base_folder = '/home/atreyee/Project_AP1XY/all_aza/C3h_from_D3h/contour_plot_CCSDT_at_MP2'

template_folder = output_base_folder+'/template'

template_file = 'opt.com'

output_MP2 = '/home/atreyee/Project_AP1XY/all_aza/C3h_from_D3h/contour_plot_at_MP2'

# Define the ranges and increments for var1 and var2
var1_start = 1.3
var1_end = 1.5
var1_increment = 0.01
var2_start = 1.3
var2_end = 1.5
var2_increment = 0.01


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
        if float(var1) >= float(var2):
            # Create the directory name
            folder_name = f'Mol_{var1:.2f}_{var2:.2f}'
            folder_path = os.path.join(output_base_folder, folder_name)

            # Create the directory if it does not exist
            os.makedirs(folder_path, exist_ok=True)

            # Read the template file
            template_subfolder = template_folder + ('_diag' if var1 == var2 else '_offdiag')
            with open(os.path.join(template_subfolder, template_file), 'r') as file:
                template_content = file.read()

            # Replace variables with actual values
            modified_content = template_content.replace('var1', str(var1)).replace('var2', str(var2))
            
            # Write the modified content to a new opt.com file in the new directory
            new_file_path = os.path.join(folder_path, template_file)
            with open(new_file_path, 'w') as new_file:
                new_file.write(modified_content)

            # Set up grep command for optimized variables
            grep_lines = "3" if var1 == var2 else "7"
            cmd = f"grep -A{grep_lines} ' Optimized variables' ../contour_plot_at_MP2/{folder_name}/opt.out | tail -{grep_lines} >> {folder_name}/opt.com"
            os.system(cmd)

            # Append basis set and method details without stars
            with open(f"{folder_name}/opt.com", "a") as opt_file:
                opt_file.write("\n\nbasis=cc-pVDZ\nhf\nccsd(t)\n")

print("All combinations have been created.")
```
#
```
import glob

# Define the folder where the CSV files are located
folder_path = './'  # Adjust this path if needed

# Get all CSV files in the folder
csv_files = glob.glob(f'{folder_path}/*.csv')

# Prepare a list to hold the formatted rows
formatted_rows = []

# Process each CSV file
for file in csv_files:
    # Read the CSV file without headers
    with open(file, 'r') as f:
        # Skip the first line (header)
        next(f)
        # Read each line and append to the list
        for line in f:
            # Remove any leading/trailing whitespace and split by whitespace
            row = line.strip().split()
            # Join the row elements with commas and add to formatted_rows
            formatted_rows.append(row)  # Store as lists for sorting

# Sort rows based on the first column (convert to float for sorting)
formatted_rows.sort(key=lambda x: float(x[0]))

# Write the sorted rows to a new CSV file with commas
with open('all.csv', 'w') as f:
    for row in formatted_rows:
        f.write(','.join(map(str, row)) + '\n')

# Calculate the maximum width of each column for alignment
max_lengths = [max(len(str(item)) for item in col) for col in zip(*formatted_rows)]

# Create a list to hold the aligned output
aligned_output = []
for row in formatted_rows:
    # Format each row with proper spacing based on max_lengths
    aligned_row = '   '.join(f'{str(item):<{max_lengths[i]}}' for i, item in enumerate(row))
    aligned_output.append(aligned_row)

# Write the aligned output to a text file
with open('aligned_output.txt', 'w') as f:
    for row in aligned_output:
        f.write(row + '\n')

print("CSV files have been merged, sorted, and written to 'all.csv'.")
print("An aligned output has been written to 'aligned_output.txt'.")
```
# 
```
import glob

# Define the folder where the CSV files are located
folder_path = './'  # Adjust this path if needed

# Get all CSV files in the folder
csv_files = glob.glob(f'{folder_path}/*.csv')

# Prepare a list to hold the formatted rows
formatted_rows = []

# Process each CSV file
for file in csv_files:
    # Read the CSV file without headers
    with open(file, 'r') as f:
        # Skip the first line (header)
        next(f)
        # Read each line and append to the list
        for line in f:
            # Remove any leading/trailing whitespace and split by whitespace
            row = line.strip().split()
            # Join the row elements with commas and add to formatted_rows
            formatted_rows.append(row)  # Store as lists for sorting

# Sort rows based on the first column (convert to float for sorting)
formatted_rows.sort(key=lambda x: float(x[0]))

# Write the sorted rows to a new CSV file with commas
with open('all.csv', 'w') as f:
    for row in formatted_rows:
        f.write(','.join(map(str, row)) + '\n')

# Calculate the maximum width of each column for alignment
max_lengths = [max(len(str(item)) for item in col) for col in zip(*formatted_rows)]

# Create a list to hold the aligned output
aligned_output = []
for row in formatted_rows:
    # Format each row with proper spacing based on max_lengths
    aligned_row = '   '.join(f'{str(item):<{max_lengths[i]}}' for i, item in enumerate(row))
    aligned_output.append(aligned_row)

# Write the aligned output to a text file
with open('aligned_output.txt', 'w') as f:
    for row in aligned_output:
        f.write(row + '\n')

print("CSV files have been merged, sorted, and written to 'all.csv'.")
print("An aligned output has been written to 'aligned_output.txt'.")
```
#
```
import os

def create_folders_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    # Open and read the file
    with open(file_path, 'r') as file:
        folder_names = file.readlines()

    # Create folders for each name in the file
    for folder_name in folder_names:
        folder_name = folder_name.strip()  # Remove any leading/trailing whitespace
        if folder_name:  # Ensure the folder name is not empty
            sanitized_folder_name = folder_name.replace(',', '_').replace('-', '_')
            try:
                os.makedirs(sanitized_folder_name, exist_ok=True)
                print(f"Created folder: {sanitized_folder_name}")
            except OSError as e:
                print(f"Error creating folder {sanitized_folder_name}: {e}")

# Specify the path to your text file
file_path = 'a.txt'

# Call the function to create folders
create_folders_from_file(file_path)
```
# 
```
import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    start_time = time.time()
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Define the size range as powers of 2 from 2^0 to 2^13
sizes = [2**i for i in range(14)]

# Initialize lists to store times
times_bubble = []
times_quicksort = []
times_mergesort = []
times_heapsort = []

for size in sizes:
    array = np.random.rand(size)
    
    # Bubble Sort
    times_bubble.append(bubble_sort(array.copy()))
    
    # Quick Sort
    start_time = time.time()
    np.sort(array.copy(), kind='quicksort')
    end_time = time.time()
    times_quicksort.append(end_time - start_time)
    
    # Merge Sort
    start_time = time.time()
    np.sort(array.copy(), kind='mergesort')
    end_time = time.time()
    times_mergesort.append(end_time - start_time)
    
    # Heap Sort
    start_time = time.time()
    np.sort(array.copy(), kind='heapsort')
    end_time = time.time()
    times_heapsort.append(end_time - start_time)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the CPU times
plt.plot(sizes, times_bubble, label='Bubble Sort', marker='o')
plt.plot(sizes, times_quicksort, label='Quick Sort', marker='o')
plt.plot(sizes, times_mergesort, label='Merge Sort', marker='o')
plt.plot(sizes, times_heapsort, label='Heap Sort', marker='o')

# Plot aN^2 and bNlog(N)
a = 1e-7  # Adjustable constant
b = 1e-6  # Adjustable constant
plt.plot(sizes, [a*(n**2) for n in sizes], label='$aN^2$', linestyle='--')
plt.plot(sizes, [b*n*np.log(n) for n in sizes], label='$bN\log(N)$', linestyle='--')

# Labeling the plot
plt.xlabel('Array Size (N)')
plt.ylabel('CPU Time (s)')
plt.title('CPU Time for Various Sorting Algorithms')
plt.legend()
plt.grid(True)
plt.xscale('log', base=2)
plt.show()

```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Constants for unit conversion
hartree2kcm = 627.509
hartree2eV = 27.2114  # Conversion from Hartree to eV

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Normalize energy values
        energy = (energy - np.min(energy)) * hartree2eV  # Convert to eV for comparison

        # Identify cases where var1 == var2 and var1 != var2
        same_condition = (var1 == var2)
        diff_condition = (var1 != var2)

        # Check for minimum energy under the conditions
        if np.any(same_condition):
            min_energy_same = np.min(energy[same_condition])
        else:
            min_energy_same = None
            print("Warning: No cases found where var1 == var2.")

        if np.any(diff_condition):
            min_energy_diff = np.min(energy[diff_condition])
        else:
            min_energy_diff = None
            print("Warning: No cases found where var1 != var2.")

        # Print energy values
        if min_energy_same is not None:
            print(f"Minimum energy (var1 == var2): {min_energy_same:.4f} eV")
        if min_energy_diff is not None:
            print(f"Minimum energy (var1 != var2): {min_energy_diff:.4f} eV")

        # Compare absolute difference and print result
        if min_energy_same is not None and min_energy_diff is not None:
            energy_difference = abs(min_energy_same - min_energy_diff)
            if energy_difference >= 0.1:
                print("Result: distortion")
            else:
                print("Result: no distortion")
        else:
            print("Comparison not possible due to missing data.")

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy, energy)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Plot contour
        fig, ax = plt.subplots(figsize=(10, 8))
        levels = np.linspace(0, np.max(energy), 50)
        cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        ax.set_xlabel("$r_1$ [$\AA$]")
        ax.set_ylabel("$r_2$ [$\AA$]")
        ax.set_title(f'Contour Plot for {filename}')
        plt.show()
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Constants for unit conversion
hartree2kcm = 627.509
hartree2eV = 27.2114  # Conversion from Hartree to eV

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Normalize energy values
        energy = (energy - np.min(energy)) * hartree2eV  # Convert to eV for comparison

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy, energy)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Plot contour
        fig, ax = plt.subplots(figsize=(10, 8))
        levels = np.linspace(0, np.max(energy), 50)
        cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        ax.set_xlabel("$r_1$ [$\AA$]")
        ax.set_ylabel("$r_2$ [$\AA$]")
        ax.set_title(f'Contour Plot for {filename}')
        plt.show()

        # Calculate minimum energy for cases where var1 == var2 and var1 != var2
        min_energy_same = np.min(energy[(var1 == var2)])
        min_energy_diff = np.min(energy[(var1 != var2)])

        # Print energy values
        print(f"Minimum energy (var1 == var2): {min_energy_same:.4f} eV")
        print(f"Minimum energy (var1 != var2): {min_energy_diff:.4f} eV")

        # Compare absolute difference and print result
        energy_difference = abs(min_energy_same - min_energy_diff)
        if energy_difference >= 0.1:
            print("Result: distortion")
        else:
            print("Result: no distortion")
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Constants for unit conversion
hartree2kcm = 627.509
hartree2eV = 27.2114  # Conversion from Hartree to eV

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Normalize energy values
        energy = (energy - np.min(energy)) * hartree2eV  # Convert to eV for comparison

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy, energy)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Plot contour
        fig, ax = plt.subplots(figsize=(10, 8))
        levels = np.linspace(0, np.max(energy), 50)
        cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        ax.set_xlabel("$r_1$ [$\AA$]")
        ax.set_ylabel("$r_2$ [$\AA$]")
        ax.set_title(f'Contour Plot for {filename}')
        plt.show()

        # Calculate minimum energy for cases where var1 == var2 and var1 != var2
        same_condition = (var1 == var2)
        diff_condition = (var1 != var2)

        if np.any(same_condition):
            min_energy_same = np.min(energy[same_condition])
            print(f"Minimum energy (var1 == var2): {min_energy_same:.4f} eV")
        else:
            min_energy_same = None
            print("No cases found where var1 == var2.")

        if np.any(diff_condition):
            min_energy_diff = np.min(energy[diff_condition])
            print(f"Minimum energy (var1 != var2): {min_energy_diff:.4f} eV")
        else:
            min_energy_diff = None
            print("No cases found where var1 != var2.")

        # Compare absolute difference and print result
        if min_energy_same is not None and min_energy_diff is not None:
            energy_difference = abs(min_energy_diff - min_energy_same)
            print(f"Energy difference: {energy_difference:.4f} eV")
            if energy_difference >= 0.1:
                print("Result: distortion")
            else:
                print("Result: no distortion")
        else:
            print("Comparison not possible due to missing data.")
```
# rosenbrock
```
import numpy as np
from scipy.optimize import minimize

# Define the Himmelblau function
def himmelblau(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Known global minima for verification
known_minima = [
    (3, 2),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126)
]

# Initial guesses
initial_guesses = [np.array([1, 1]), np.array([1, -1]), np.array([-1, 1]), np.array([-1, -1])]

# Store results
results = []

for initial_guess in initial_guesses:
    # Optimize using BFGS
    result = minimize(himmelblau, initial_guess, method='BFGS', options={'disp': True})
    results.append((result.x, result.fun, result.nit))

# Display the results
for i, (x, fun_val, iterations) in enumerate(results):
    print(f"Initial guess {initial_guesses[i]}: Minimum found at {x} with function value {fun_val} in {iterations} iterations.")

# Verify against known minima
for minimum in known_minima:
    value = himmelblau(minimum)
    print(f"Known minimum {minimum} has function value {value}.")
```
#
```
import numpy as np
from scipy.optimize import minimize

# Define the function
def new_function(v):
    x, y, z, w = v
    return (x + 10*y)**2 + 5*(z - w)**2 + (y - 2*z)**4 + 100*(x - w)**4

# Initial guess for the variables
initial_guess = np.array([1, 1, 1, 1])

# Optimization using the 'BFGS' algorithm to find the minimum and count iterations
result = minimize(new_function, initial_guess, method='BFGS', options={'disp': True})

# Display the result
print("Minimum found at:", result.x)
print("Number of iterations:", result.nit)
```
#
```
import numpy as np
from scipy.optimize import minimize

# Define the Himmelblau function
def himmelblau(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Initial guesses
initial_guesses = [np.array([1, 1]), np.array([1, -1]), np.array([-1, 1]), np.array([-1, -1])]

# Store results
results = []

for initial_guess in initial_guesses:
    # Optimize using BFGS
    result = minimize(himmelblau, initial_guess, method='BFGS', options={'disp': True})
    results.append((result.x, result.fun, result.nit))

# Display the results
for i, (x, fun_val, iterations) in enumerate(results):
    print(f"Initial guess {initial_guesses[i]}: Minimum found at {x} with function value {fun_val} in {iterations} iterations.")
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Constants for unit conversion
hartree2kcm = 627.509
hartree2eV = 27.2114  # Conversion from Hartree to eV

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Normalize energy values
        energy = (energy - np.min(energy)) * hartree2eV  # Convert to eV for comparison

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy, energy)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Plot contour
        fig, ax = plt.subplots(figsize=(10, 8))
        levels = np.linspace(0, np.max(energy), 50)
        cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        ax.set_xlabel("$r_1$ [$\AA$]")
        ax.set_ylabel("$r_2$ [$\AA$]")
        ax.set_title(f'Contour Plot for {filename}')
        plt.show()

        # Calculate minimum energy for cases where var1 == var2 and var1 != var2
        same_condition = (var1 == var2)
        diff_condition = (var1 != var2)

        if np.any(same_condition):
            min_energy_same = np.min(energy[same_condition])
            min_index_same = np.where(same_condition & (energy == min_energy_same))[0][0]
            print(f"Minimum energy (var1 == var2): {min_energy_same:.4f} eV at (var1, var2) = ({var1[min_index_same]:.4f}, {var2[min_index_same]:.4f})")
        else:
            min_energy_same = None
            print("No cases found where var1 == var2.")

        if np.any(diff_condition):
            min_energy_diff = np.min(energy[diff_condition])
            min_index_diff = np.where(diff_condition & (energy == min_energy_diff))[0][0]
            print(f"Minimum energy (var1 != var2): {min_energy_diff:.4f} eV at (var1, var2) = ({var1[min_index_diff]:.4f}, {var2[min_index_diff]:.4f})")
        else:
            min_energy_diff = None
            print("No cases found where var1 != var2.")

        # Compare absolute difference and print result
        if min_energy_same is not None and min_energy_diff is not None:
            energy_difference = abs(min_energy_diff - min_energy_same)
            print(f"Energy difference: {energy_difference:.4f} eV")
            if energy_difference >= 0.1:
                print("Result: distortion")
            else:
                print("Result: no distortion")
        else:
            print("Comparison not possible due to missing data.")
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

hartree2kcm = 627.509
kcal2ev = 0.0433641  # Conversion factor from kcal/mol to eV
alpha = 0.4  # Transparency for the colormap
cmap = plt.cm.viridis
cmap_colors = cmap(np.arange(cmap.N))
cmap_colors[:, -1] = alpha
cmap_alpha = ListedColormap(cmap_colors)
levels = np.linspace(-1, 30, 30)

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        # Read the CSV file
        data = pd.read_csv(filename)
        
        # Extract x, y, and z values
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
        z = data.iloc[:, 2]
        
        # Normalize energy data and convert to kcal/mol
        z = z - np.min(z)
        z = z * hartree2kcm
        
        # Convert z from kcal/mol to eV
        z_ev = z * kcal2ev
        
        # Extend the grid to cover the range of data
        xi = np.linspace(x.min(), x.max(), 1050)
        yi = np.linspace(y.min(), y.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        
        # Use `method='linear'` if `cubic` is causing issues with edges
        zi = griddata((x, y), z, (xi, yi), method='linear')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cp = plt.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        plt.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        
        plt.xlabel("$r_1$ [$\AA$]")
        plt.ylabel("$r_2$ [$\AA$]")
        plt.title(f'Contour Plot for {filename}')
        plt.show()
        
        # Find minimum energy for var1 == var2
        min_equal_energy_ev = z_ev[(x == y)].min()
        
        # Find minimum energy for var1 != var2
        min_unequal_energy_ev = z_ev[(x != y)].min()
        
        # Print minimum energy values
        print(f'File: {filename}')
        print(f'Minimum energy (var1 == var2): {min_equal_energy_ev:.4f} eV')
        print(f'Minimum energy (var1 != var2): {min_unequal_energy_ev:.4f} eV')
        
        # Check absolute difference in eV
        abs_diff_ev = abs(min_equal_energy_ev - min_unequal_energy_ev)
        if abs_diff_ev >= 0.1:
            print('Distortion')
        else:
            print('No distortion')
        print('-' * 50)
```
#
```
import os

def create_folders_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    # Open and read the file
    with open(file_path, 'r') as file:
        folder_names = file.readlines()

    # Create folders for each name in the file
    for folder_name in folder_names:
        folder_name = folder_name.strip()  # Remove any leading/trailing whitespace
        if folder_name:  # Ensure the folder name is not empty
            sanitized_folder_name = folder_name.replace(',', '_').replace('-', '_')
            try:
                os.makedirs(sanitized_folder_name, exist_ok=True)
                print(f"Created folder: {sanitized_folder_name}")
            except OSError as e:
                print(f"Error creating folder {sanitized_folder_name}: {e}")

# Specify the path to your text file
file_path = 'a.txt'

# Call the function to create folders
create_folders_from_file(file_path)
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Updated constant for unit conversion
hartree2kJmol = 2625.5  # Conversion from Hartree to kJ/mol

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Convert all energy values from Hartree to kJ/mol
        energy_kjmol = energy * hartree2kJmol

        # Normalize energy values in kJ/mol
        energy_kjmol = energy_kjmol - np.min(energy_kjmol)

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy_kjmol, energy_kjmol)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Plot contour
        fig, ax = plt.subplots(figsize=(10, 8))
        levels = np.linspace(0, np.max(energy_kjmol), 50)
        cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        ax.set_xlabel("$r_1$ [$\AA$]")
        ax.set_ylabel("$r_2$ [$\AA$]")
        ax.set_title(f'Contour Plot for {filename}')
        plt.show()

        # Calculate minimum energy for cases where var1 == var2 and var1 != var2
        same_condition = (var1 == var2)
        diff_condition = (var1 != var2)

        if np.any(same_condition):
            min_energy_same = np.min(energy_kjmol[same_condition])
            min_index_same = np.where(same_condition & (energy_kjmol == min_energy_same))[0][0]
            print(f"Minimum energy (var1 == var2): {min_energy_same:.4f} kJ/mol at (var1, var2) = ({var1[min_index_same]:.4f}, {var2[min_index_same]:.4f})")
        else:
            min_energy_same = None
            print("No cases found where var1 == var2.")

        if np.any(diff_condition):
            min_energy_diff = np.min(energy_kjmol[diff_condition])
            min_index_diff = np.where(diff_condition & (energy_kjmol == min_energy_diff))[0][0]
            var1_diff = var1[min_index_diff]
            var2_diff = var2[min_index_diff]
            diff_var1_var2 = abs(var1_diff - var2_diff)
            print(f"Minimum energy (var1 != var2): {min_energy_diff:.4f} kJ/mol at (var1, var2) = ({var1_diff:.4f}, {var2_diff:.4f})")
            print(f"Difference between var1 and var2 at min energy: {diff_var1_var2:.4f}")
        else:
            min_energy_diff = None
            diff_var1_var2 = None
            print("No cases found where var1 != var2.")

        # Compare absolute energy difference and apply the second condition
        if min_energy_same is not None and min_energy_diff is not None:
            energy_difference = abs(min_energy_diff - min_energy_same)
            print(f"Energy difference: {energy_difference:.4f} kJ/mol")
            if energy_difference >= 0.3 and (diff_var1_var2 is not None and diff_var1_var2 > 0.03):
                print("Result: distortion")
            else:
                print("Result: no distortion")
        else:
            print("Comparison not possible due to missing data.")
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Constants for unit conversion
hartree2kcm = 627.509  # Conversion from Hartree to kJ/mol

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Normalize energy values (still in Hartree units)
        energy = energy - np.min(energy)

        # Convert energy values to kJ/mol
        energy_kjmol = energy * hartree2kcm

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy_kjmol, energy_kjmol)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Plot contour
        fig, ax = plt.subplots(figsize=(10, 8))
        levels = np.linspace(0, np.max(energy_kjmol), 50)
        cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
        plt.colorbar(cp)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        ax.set_xlabel("$r_1$ [$\AA$]")
        ax.set_ylabel("$r_2$ [$\AA$]")
        ax.set_title(f'Contour Plot for {filename}')
        plt.show()

        # Calculate minimum energy for cases where var1 == var2 and var1 != var2
        same_condition = (var1 == var2)
        diff_condition = (var1 != var2)

        if np.any(same_condition):
            min_energy_same = np.min(energy_kjmol[same_condition])
            min_index_same = np.where(same_condition & (energy_kjmol == min_energy_same))[0][0]
            print(f"Minimum energy (var1 == var2): {min_energy_same:.4f} kJ/mol at (var1, var2) = ({var1[min_index_same]:.4f}, {var2[min_index_same]:.4f})")
        else:
            min_energy_same = None
            print("No cases found where var1 == var2.")

        if np.any(diff_condition):
            min_energy_diff = np.min(energy_kjmol[diff_condition])
            min_index_diff = np.where(diff_condition & (energy_kjmol == min_energy_diff))[0][0]
            print(f"Minimum energy (var1 != var2): {min_energy_diff:.4f} kJ/mol at (var1, var2) = ({var1[min_index_diff]:.4f}, {var2[min_index_diff]:.4f})")
        else:
            min_energy_diff = None
            print("No cases found where var1 != var2.")

        # Compare absolute difference and print result
        if min_energy_same is not None and min_energy_diff is not None:
            energy_difference = abs(min_energy_diff - min_energy_same)
            print(f"Energy difference: {energy_difference:.4f} kJ/mol")
            if energy_difference >= 0.3:
                print("Result: distortion")
            else:
                print("Result: no distortion")
        else:
            print("Comparison not possible due to missing data.")
```
#
```
import glob

# Define the folder where the CSV files are located
folder_path = './'  # Adjust this path if needed

# Get all CSV files in the folder
csv_files = glob.glob(f'{folder_path}/*.csv')

# Prepare a list to hold the formatted rows
formatted_rows = []

# Process each CSV file
for file in csv_files:
    # Read the CSV file without headers
    with open(file, 'r') as f:
        # Skip the first line (header)
        next(f)
        # Read each line and append to the list
        for line in f:
            # Remove any leading/trailing whitespace and split by whitespace
            row = line.strip().split()
            # Join the row elements with commas and add to formatted_rows
            formatted_rows.append(row)  # Store as lists for sorting

# Sort rows based on the first column (convert to float for sorting)
formatted_rows.sort(key=lambda x: float(x[0]))

# Write the sorted rows to a new CSV file with commas
with open('all.csv', 'w') as f:
    for row in formatted_rows:
        f.write(','.join(map(str, row)) + '\n')

# Calculate the maximum width of each column for alignment
max_lengths = [max(len(str(item)) for item in col) for col in zip(*formatted_rows)]

# Create a list to hold the aligned output
aligned_output = []
for row in formatted_rows:
    # Format each row with proper spacing based on max_lengths
    aligned_row = '   '.join(f'{str(item):<{max_lengths[i]}}' for i, item in enumerate(row))
    aligned_output.append(aligned_row)

# Write the aligned output to a text file
with open('aligned_output.txt', 'w') as f:
    for row in aligned_output:
        f.write(row + '\n')

print("CSV files have been merged, sorted, and written to 'all.csv'.")
print("An aligned output has been written to 'aligned_output.txt'.")
```
#
```
import os
import shutil

# Define the paths
extrapolate_folder = './extrapolate'
adc2_folder = './adc2'

# Create the adc2 folder if it doesn't exist
if not os.path.exists(adc2_folder):
    os.makedirs(adc2_folder)

# Loop through each folder inside extrapolate
for folder_name in os.listdir(extrapolate_folder):
    folder_path = os.path.join(extrapolate_folder, folder_name)
    
    # Check if it is a directory
    if os.path.isdir(folder_path):
        xyz_file = os.path.join(folder_path, 'test.xyz')
        
        # Read the test.xyz file, skipping the first two lines
        with open(xyz_file, 'r') as xyz:
            lines = xyz.readlines()[2:]
        
        # Prepare the Q-Chem input template
        qchem_input = '''$molecule
  0  1

'''
        # Add coordinates
        qchem_input += ''.join(lines)
        qchem_input += '''$end

$rem
jobtype             sp
method              adc(2)
basis               cc-pVTZ
aux_basis           rimp2-cc-pVTZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
'''
        
        # Create the corresponding folder inside adc2
        new_folder = os.path.join(adc2_folder, folder_name)
        os.makedirs(new_folder, exist_ok=True)
        
        # Write the all.com file inside the new folder
        allcom_file = os.path.join(new_folder, 'all.com')
        with open(allcom_file, 'w') as allcom:
            allcom.write(qchem_input)

print("Files created successfully!")
```
#
```import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = 'path_to_your_folder'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read all CSV files into a list of DataFrames (with headers)
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Sort the merged DataFrame based on the first column
merged_df = merged_df.sort_values(by=merged_df.columns[0], ascending=True)

# Save the merged and sorted DataFrame to a new CSV file
merged_df.to_csv(os.path.join(folder_path, 'merged_sorted.csv'), index=False)

print("CSV files from the folder merged and sorted successfully.")
```
#
```
import csv

# Read b.csv into a dictionary for easy lookup, skipping the header
b_data = {}
with open('b.csv', 'r') as b_file:
    b_reader = csv.reader(b_file)
    next(b_reader)  # Skip the header
    for row in b_reader:
        if row:
            b_data[row[0].strip()] = row[1:4]  # Store the 2nd, 3rd, and 4th columns for each molecule name

# Process a.csv, skipping the header, and merge the data from b.csv
merged_rows = []
with open('a.csv', 'r') as a_file:
    a_reader = csv.reader(a_file)
    header = next(a_reader)  # Read the header
    merged_header = header + ["Column7", "Column8", "Column9"]  # Add new column headers
    merged_rows.append(merged_header)

    for row in a_reader:
        if row:
            mol_name = row[3].strip()  # Get the molecule name from the 4th column
            b_values = b_data.get(mol_name, ["", "", ""])  # Get corresponding values or empty if not found
            merged_rows.append(row + b_values)  # Add the values as the 7th, 8th, and 9th columns

# Write the merged data to a new CSV file
with open('merged.csv', 'w', newline='') as merged_file:
    merged_writer = csv.writer(merged_file)
    merged_writer.writerows(merged_rows)

print("Merging completed! Check the merged.csv file.")
```
#
```
import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = 'path_to_your_folder'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Iterate through each file and check for non-numeric values in the first column
non_numeric_info = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Check if the first column has non-numeric values
    non_numeric_rows = df[~df[df.columns[0]].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna()]
    
    if not non_numeric_rows.empty:
        # Record the file name and the rows with non-numeric values
        non_numeric_info.append((file, non_numeric_rows))

# Print the non-numeric information
if non_numeric_info:
    print("Non-numeric values found in the following CSV files:")
    for file, rows in non_numeric_info:
        print(f"\nFile: {file}")
        print("Rows with non-numeric values in the first column:")
        print(rows)
else:
    print("All values in the first column of each CSV file are numeric.")
```
#
```
import pandas as pd

# Load your DataFrame
merged_df = pd.read_csv("your_file.csv")

# Try to convert the first column to numeric, setting errors='coerce' to convert non-numeric values to NaN
merged_df[merged_df.columns[0]] = pd.to_numeric(merged_df[merged_df.columns[0]], errors='coerce')

# Sort the DataFrame by the first column, excluding NaN values
merged_df = merged_df.sort_values(by=merged_df.columns[0], ascending=True).dropna()

# Save the sorted DataFrame or continue processing
merged_df.to_csv("sorted_file.csv", index=False)
```
#
```import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/mnt/data/gaps.csv'  # Adjust to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3].str.replace('_', ',')  # Fourth column, replace underscores with commas

# Define the x positions based on the sets
total_rows = len(gap_values)
x_positions = []
group_counts = [1, 4, 4, 4, 1, 1]  # Adjust the groupings as needed

if sum(group_counts) != total_rows:
    raise ValueError("The sum of group_counts must equal the number of rows in the data.")

x = 1  # Starting x position
for count in group_counts:
    x_positions.extend([x] * count)
    x += 1

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for better aesthetics
ax.set_xlim(0.5, max(x_positions) + 0.5)
ax.set_ylim(-0.35, 0.1)  # Adjust y-axis range for more space
ax.set_ylabel('Gap Value (eV)', fontsize=12)
ax.set_xlabel('Sets of Molecules', fontsize=12)
ax.set_title('Energy Gaps of Molecules', fontsize=14, fontweight='bold')

# Customize the plot's appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Plot the lines for each gap value at the corresponding x position
for i in range(len(gap_values)):
    # Draw a horizontal line for each gap value
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black', linewidth=1)
    
    # Adjust label position to avoid overlap for specific cases
    label_offset = 0
    if i == 1:  # Example: Move the label up for the second line (index 1)
        label_offset = 0.03  # Move this label upward
    elif i == 2:  # Example: Move the label down for the third line (index 2)
        label_offset = -0.03  # Move this label downward

    # Annotate the gap value on the left side of the line
    ax.text(x_positions[i] - 0.15, gap_values[i] + label_offset, f'{gap_values[i]}', ha='right', fontsize=9, color='darkred')
    
    # Annotate the molecule name on the right side of the line
    ax.text(x_positions[i] + 0.15, gap_values[i] + label_offset, molecule_names[i], ha='left', fontsize=9, color='darkblue')

# Improve the overall aesthetics
plt.tight_layout()
plt.show()
```
#
```
import pandas as pd

# Read the CSV file (assuming no headers; update header=None if needed)
file_path = "your_file.csv"  # Replace with your actual file path
csv_data = pd.read_csv(file_path, header=None)

# Extract columns based on the mapping
table_data = pd.DataFrame({
    "Col1": csv_data.iloc[:, 3],
    "Col2": csv_data.iloc[:, 4],
    "Col3": csv_data.iloc[:, 1],
    "Col4": csv_data.iloc[:, 2],
    "Col5": csv_data.iloc[:, 0].str.replace("_", ","),  # Replace underscores in molecule names
    "Col6": csv_data.iloc[:, 6],
    "Col7": csv_data.iloc[:, 7],
    "Col8": csv_data.iloc[:, 8]
})

# Assign headers from the original CSV
headers = [
    str(csv_data.columns[3]),
    str(csv_data.columns[4]),
    str(csv_data.columns[1]),
    str(csv_data.columns[2]),
    str(csv_data.columns[0]),
    str(csv_data.columns[6]),
    str(csv_data.columns[7]),
    str(csv_data.columns[8])
]

# Generate the table in LaTeX format
latex_table = " & ".join(headers) + " \\\\\n"  # Add header row
latex_table += "\\hline\n"  # Add a horizontal line
for _, row in table_data.iterrows():
    latex_table += " & ".join(map(str, row)) + " \\\\\n"

# Print the result
print(latex_table)
```
#
```
import os
import shutil

def copy_and_create_folders(extrapolate_path, opt_path):
    # Check if the extrapolate folder exists
    if not os.path.exists(extrapolate_path):
        print(f"Extrapolate folder '{extrapolate_path}' does not exist.")
        return
    
    # Ensure opt folder exists
    if not os.path.exists(opt_path):
        print(f"Opt folder '{opt_path}' does not exist.")
        return
    
    opt_com_path = os.path.join(opt_path, 'opt.com')
    
    # Check if opt.com exists
    if not os.path.isfile(opt_com_path):
        print(f"'opt.com' file not found in the opt folder.")
        return
    
    # Iterate through each folder in the extrapolate folder
    for folder in os.listdir(extrapolate_path):
        extrapolate_subfolder = os.path.join(extrapolate_path, folder)
        if os.path.isdir(extrapolate_subfolder):  # Ensure it is a folder
            test_xyz_path = os.path.join(extrapolate_subfolder, 'test.xyz')
            
            # Check if test.xyz exists
            if os.path.isfile(test_xyz_path):
                # Create a corresponding folder in the opt folder
                opt_subfolder = os.path.join(opt_path, folder)
                os.makedirs(opt_subfolder, exist_ok=True)
                
                # Copy test.xyz to the new folder
                shutil.copy(test_xyz_path, opt_subfolder)
                print(f"Copied '{test_xyz_path}' to '{opt_subfolder}'.")
                
                # Copy opt.com to the new folder
                shutil.copy(opt_com_path, opt_subfolder)
                print(f"Copied '{opt_com_path}' to '{opt_subfolder}'.")
            else:
                print(f"'test.xyz' not found in '{extrapolate_subfolder}'.")
    print("Process completed.")

# Define paths
extrapolate_folder = 'path/to/extrapolate'  # Replace with the path to the extrapolate folder
opt_folder = 'path/to/opt'  # Replace with the path to the opt folder

# Call the function
copy_and_create_folders(extrapolate_folder, opt_folder)
```
#
```import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/mnt/data/gaps.csv'  # Adjust to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3].str.replace('_', ',')  # Fourth column, replace underscores with commas

# Define the x positions based on the sets
total_rows = len(gap_values)
x_positions = []
group_counts = [1, 4, 4, 4, 1, 1]  # Adjust the groupings as needed

if sum(group_counts) != total_rows:
    raise ValueError("The sum of group_counts must equal the number of rows in the data.")

x = 1  # Starting x position
for count in group_counts:
    x_positions.extend([x] * count)
    x += 1

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for better aesthetics
ax.set_xlim(0.5, max(x_positions) + 0.5)
ax.set_ylim(-0.35, 0.1)  # Adjust y-axis range for more space
ax.set_ylabel('Gap Value (eV)', fontsize=12)
ax.set_xlabel('Sets of Molecules', fontsize=12)
ax.set_title('Energy Gaps of Molecules', fontsize=14, fontweight='bold')

# Customize the plot's appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Plot the lines for each gap value at the corresponding x position
for i in range(len(gap_values)):
    # Draw a horizontal line for each gap value
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black', linewidth=1)
    
    # Adjust label position to avoid overlap for specific cases
    label_offset = 0
    if i == 1:  # Example: Move the label up for the second line (index 1)
        label_offset = 0.03  # Move this label upward
    elif i == 2:  # Example: Move the label down for the third line (index 2)
        label_offset = -0.03  # Move this label downward

    # Annotate the gap value on the left side of the line
    ax.text(x_positions[i] - 0.15, gap_values[i] + label_offset, f'{gap_values[i]}', ha='right', fontsize=9, color='darkred')
    
    # Annotate the molecule name on the right side of the line
    ax.text(x_positions[i] + 0.15, gap_values[i] + label_offset, molecule_names[i], ha='left', fontsize=9, color='darkblue')

# Improve the overall aesthetics
plt.tight_layout()
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/mnt/data/gaps.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3]  # Fourth column for molecule names

# Setup the x positions for each line
x_positions = []
x = 1  # Start x position
group_count = 0

for index in range(len(gap_values)):
    if group_count == 0:
        x_positions.append(x)
        x += 1
        group_count += 1
    elif group_count < 4:
        x_positions.append(x)
        x += 0.5  # Adjust spacing
        group_count += 1
    else:
        x_positions.append(x)
        x += 1
        group_count = 0

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, max(x_positions) + 1)
ax.set_ylim(-0.3, 0.05)
ax.set_ylabel('Gap Value (eV)')
ax.set_xlabel('Molecule Index')

# Plot the lines
for i in range(len(gap_values)):
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black')
    ax.text(x_positions[i], gap_values[i] + 0.01, f'{gap_values[i]:.2f}', ha='center', fontsize=8)
    ax.text(x_positions[i], -0.35, molecule_names[i], ha='center', fontsize=8, rotation=90)

# Add gridlines and adjust layout
ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()

# Show the plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/home/atreyee/project/Project_AP1XY/ADC2_CC2_data/STG_trends/gaps.csv'  # Adjust to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3].str.replace('_', ',')  # Fourth column, replace underscores with commas

# Define the x positions based on the sets
total_rows = len(gap_values)
x_positions = []
group_counts = [1, 1, 4, 4, 4, 1, 1]  # Adjust the groupings as needed

if sum(group_counts) != total_rows:
    raise ValueError("The sum of group_counts must equal the number of rows in the data.")

x = 1  # Starting x position
for count in group_counts:
    x_positions.extend([x] * count)
    x += 1

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for better aesthetics
ax.set_xlim(0.5, max(x_positions) + 0.5)
ax.set_ylim(-0.35, 0.1)  # Adjust y-axis range for more space
ax.set_ylabel('Gap Value (eV)', fontsize=12)
ax.set_xlabel('Sets of Molecules', fontsize=12)
ax.set_title('Energy Gaps of Molecules', fontsize=14, fontweight='bold')

# Customize the plot's appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Plot the lines for each gap value at the corresponding x position
for i in range(len(gap_values)):
    # Draw a horizontal line for each gap value
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black', linewidth=1)
    
    # Adjust label position to avoid overlap
    label_offset = 0
    if i > 0 and abs(gap_values[i] - gap_values[i - 1]) < 0.02:
        # Move labels further apart if values are very close
        if gap_values[i] > gap_values[i - 1]:
            label_offset = 0.03  # Upward adjustment for current label
        else:
            label_offset = -0.03  # Downward adjustment for current label

    # Annotate the gap value on the left side of the line
    ax.text(x_positions[i] - 0.15, gap_values[i] + label_offset, f'{gap_values[i]}', ha='right', fontsize=9, color='darkred')
    
    # Annotate the molecule name on the right side of the line
    ax.text(x_positions[i] + 0.15, gap_values[i] + label_offset, molecule_names[i], ha='left', fontsize=9, color='darkblue')

# Remove the gridlines
ax.grid(False)

# Save the plot as a PDF
plt.tight_layout()
plt.savefig('/home/atreyee/project/Project_AP1XY/ADC2_CC2_data/STG_trends/gaps_plot.pdf', format='pdf')

# Display the plot
plt.show()
```
#
```
import os
import shutil

def copy_and_create_folders(extrapolate_path, opt_path):
    # Check if the extrapolate folder exists
    if not os.path.exists(extrapolate_path):
        print(f"Extrapolate folder '{extrapolate_path}' does not exist.")
        return
    
    # Ensure opt folder exists
    if not os.path.exists(opt_path):
        print(f"Opt folder '{opt_path}' does not exist.")
        return
    
    opt_com_path = os.path.join(opt_path, 'opt.com')
    
    # Check if opt.com exists
    if not os.path.isfile(opt_com_path):
        print(f"'opt.com' file not found in the opt folder.")
        return
    
    # Iterate through each folder in the extrapolate folder
    for folder in os.listdir(extrapolate_path):
        extrapolate_subfolder = os.path.join(extrapolate_path, folder)
        if os.path.isdir(extrapolate_subfolder):  # Ensure it is a folder
            test_xyz_path = os.path.join(extrapolate_subfolder, 'test.xyz')
            
            # Check if test.xyz exists
            if os.path.isfile(test_xyz_path):
                # Create a corresponding folder in the opt folder
                opt_subfolder = os.path.join(opt_path, folder)
                os.makedirs(opt_subfolder, exist_ok=True)
                
                # Copy test.xyz to the new folder
                shutil.copy(test_xyz_path, opt_subfolder)
                print(f"Copied '{test_xyz_path}' to '{opt_subfolder}'.")
                
                # Copy opt.com to the new folder
                shutil.copy(opt_com_path, opt_subfolder)
                print(f"Copied '{opt_com_path}' to '{opt_subfolder}'.")
            else:
                print(f"'test.xyz' not found in '{extrapolate_subfolder}'.")
    print("Process completed.")

# Define paths
extrapolate_folder = 'path/to/extrapolate'  # Replace with the path to the extrapolate folder
opt_folder = 'path/to/opt'  # Replace with the path to the opt folder

# Call the function
copy_and_create_folders(extrapolate_folder, opt_folder)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/mnt/data/gaps.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3]  # Fourth column for molecule names

# Setup the x positions for each line
x_positions = []
x = 1  # Start x position
group_count = 0

for index in range(len(gap_values)):
    if group_count == 0:
        x_positions.append(x)
        x += 1
        group_count += 1
    elif group_count < 4:
        x_positions.append(x)
        x += 0.5  # Adjust spacing
        group_count += 1
    else:
        x_positions.append(x)
        x += 1
        group_count = 0

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, max(x_positions) + 1)
ax.set_ylim(-0.3, 0.05)
ax.set_ylabel('Gap Value (eV)')
ax.set_xlabel('Molecule Index')

# Plot the lines
for i in range(len(gap_values)):
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black')
    ax.text(x_positions[i], gap_values[i] + 0.01, f'{gap_values[i]:.2f}', ha='center', fontsize=8)
    ax.text(x_positions[i], -0.35, molecule_names[i], ha='center', fontsize=8, rotation=90)

# Add gridlines and adjust layout
ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()

# Show the plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/home/atreyee/project/Project_AP1XY/ADC2_CC2_data/STG_trends/gaps.csv'  # Adjust to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3].str.replace('_', ',')  # Fourth column, replace underscores with commas

# Define the x positions based on the sets
total_rows = len(gap_values)
x_positions = []
group_counts = [1, 1, 4, 4, 4, 1, 1]  # Adjust the groupings as needed

if sum(group_counts) != total_rows:
    raise ValueError("The sum of group_counts must equal the number of rows in the data.")

x = 1  # Starting x position
for count in group_counts:
    x_positions.extend([x] * count)
    x += 1

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for better aesthetics
ax.set_xlim(0.5, max(x_positions) + 0.5)
ax.set_ylim(-0.35, 0.1)  # Adjust y-axis range for more space
ax.set_ylabel('Gap Value (eV)', fontsize=12)
ax.set_xlabel('Sets of Molecules', fontsize=12)
ax.set_title('Energy Gaps of Molecules', fontsize=14, fontweight='bold')

# Customize the plot's appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Plot the lines for each gap value at the corresponding x position
for i in range(len(gap_values)):
    # Draw a horizontal line for each gap value
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black', linewidth=1)
    
    # Adjust label position to avoid overlap
    label_offset = 0
    if i > 0 and abs(gap_values[i] - gap_values[i - 1]) < 0.02:
        # Move labels further apart if values are very close
        if gap_values[i] > gap_values[i - 1]:
            label_offset = 0.03  # Upward adjustment for current label
        else:
            label_offset = -0.03  # Downward adjustment for current label

    # Annotate the gap value on the left side of the line
    ax.text(x_positions[i] - 0.15, gap_values[i] + label_offset, f'{gap_values[i]}', ha='right', fontsize=9, color='darkred')
    
    # Annotate the molecule name on the right side of the line
    ax.text(x_positions[i] + 0.15, gap_values[i] + label_offset, molecule_names[i], ha='left', fontsize=9, color='darkblue')

# Remove the gridlines
ax.grid(False)

# Save the plot as a PDF
plt.tight_layout()
plt.savefig('/home/atreyee/project/Project_AP1XY/ADC2_CC2_data/STG_trends/gaps_plot.pdf', format='pdf')

# Display the plot
plt.show()
```
#
```
import os
import shutil

def copy_and_create_folders(extrapolate_path, opt_path):
    # Check if the extrapolate folder exists
    if not os.path.exists(extrapolate_path):
        print(f"Extrapolate folder '{extrapolate_path}' does not exist.")
        return
    
    # Ensure opt folder exists
    if not os.path.exists(opt_path):
        print(f"Opt folder '{opt_path}' does not exist.")
        return
    
    opt_com_path = os.path.join(opt_path, 'opt.com')
    
    # Check if opt.com exists
    if not os.path.isfile(opt_com_path):
        print(f"'opt.com' file not found in the opt folder.")
        return
    
    # Iterate through each folder in the extrapolate folder
    for folder in os.listdir(extrapolate_path):
        extrapolate_subfolder = os.path.join(extrapolate_path, folder)
        if os.path.isdir(extrapolate_subfolder):  # Ensure it is a folder
            test_xyz_path = os.path.join(extrapolate_subfolder, 'test.xyz')
            
            # Check if test.xyz exists
            if os.path.isfile(test_xyz_path):
                # Create a corresponding folder in the opt folder
                opt_subfolder = os.path.join(opt_path, folder)
                os.makedirs(opt_subfolder, exist_ok=True)
                
                # Copy test.xyz to the new folder
                shutil.copy(test_xyz_path, opt_subfolder)
                print(f"Copied '{test_xyz_path}' to '{opt_subfolder}'.")
                
                # Copy opt.com to the new folder
                shutil.copy(opt_com_path, opt_subfolder)
                print(f"Copied '{opt_com_path}' to '{opt_subfolder}'.")
            else:
                print(f"'test.xyz' not found in '{extrapolate_subfolder}'.")
    print("Process completed.")

# Define paths
extrapolate_folder = 'path/to/extrapolate'  # Replace with the path to the extrapolate folder
opt_folder = 'path/to/opt'  # Replace with the path to the opt folder

# Call the function
copy_and_create_folders(extrapolate_folder, opt_folder)
```
# 
```
import pandas as pd

# Read the CSV file (assuming no headers; update header=None if needed)
file_path = "your_file.csv"  # Replace with your actual file path
csv_data = pd.read_csv(file_path, header=None)

# Extract columns based on the mapping
table_data = pd.DataFrame({
    "Col1": csv_data.iloc[:, 3],
    "Col2": csv_data.iloc[:, 4],
    "Col3": csv_data.iloc[:, 1],
    "Col4": csv_data.iloc[:, 2],
    "Col5": csv_data.iloc[:, 0].str.replace("_", ","),  # Replace underscores in molecule names
    "Col6": csv_data.iloc[:, 6],
    "Col7": csv_data.iloc[:, 7],
    "Col8": csv_data.iloc[:, 8]
})

# Assign headers from the original CSV
headers = [
    str(csv_data.columns[3]),
    str(csv_data.columns[4]),
    str(csv_data.columns[1]),
    str(csv_data.columns[2]),
    str(csv_data.columns[0]),
    str(csv_data.columns[6]),
    str(csv_data.columns[7]),
    str(csv_data.columns[8])
]

# Generate the table in LaTeX format
latex_table = " & ".join(headers) + " \\\\\n"  # Add header row
latex_table += "\\hline\n"  # Add a horizontal line
for _, row in table_data.iterrows():
    latex_table += " & ".join(map(str, row)) + " \\\\\n"

# Print the result
print(latex_table)
```
#
```
import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch, Rectangle

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Function to draw energy levels
def draw_energy_level(y, label=None):
    ax.plot([0.1, 0.9], [y, y], 'k', lw=2)
    if label:
        ax.text(0.92, y, label, va='center', fontsize=10)

# Function to draw vertical arrows
def draw_arrow(x, y_start, y_end, label=None, color='black', style='solid'):
    arrow = FancyArrowPatch((x, y_start), (x, y_end), arrowstyle='->', 
                            color=color, lw=1.5, linestyle=style)
    ax.add_patch(arrow)
    if label:
        ax.text(x + 0.02, (y_start + y_end) / 2, label, va='center', fontsize=8, rotation=90)

# Draw energy levels for S0, S1, S2, T1, T2, T3
draw_energy_level(1, 'S0')
draw_energy_level(2.5, 'S1')
draw_energy_level(4, 'S2')
draw_energy_level(1.5, 'T1')
draw_energy_level(3, 'T2')
draw_energy_level(4.5, 'T3')

# Draw arrows and labels
draw_arrow(0.2, 1, 2.5, 'k_A\nAbsorption')
draw_arrow(0.25, 2.5, 1, 'k_F\nFluorescence', color='green')
draw_arrow(0.3, 2.5, 1.5, 'k_ISC\nIntersystem crossing', style='dashed')
draw_arrow(0.35, 1.5, 1, 'k_Ph\nPhosphorescence', color='purple')

# Additional components like vibrational relaxation and non-radiative relaxation
ax.text(0.1, 3.25, 'Vibrational relaxation', fontsize=8)
ax.text(0.1, 1.75, 'Non-radiative relaxation', fontsize=8)

# Draw T-T absorption arrows
draw_arrow(0.4, 1.5, 3, 'T-T\nAbsorption', color='gray', style='dashed')

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0.5, 5)

# Show the diagram
plt.tight_layout()
plt.show()
```
#
```
import pandas as pd

# Load the CSV files
csv_16_columns = pd.read_csv('csv_16_columns.csv', header=None)
csv_1_column = pd.read_csv('csv_1_column.csv', header=None)

# Ensure the single column DataFrame has the same number of rows as the 16 columns DataFrame
if len(csv_16_columns) != len(csv_1_column):
    raise ValueError("The number of rows in the two CSV files does not match.")

# Add the single column from the second CSV to the first CSV
csv_merged = pd.concat([csv_16_columns, csv_1_column], axis=1)

# Save the merged DataFrame to a new CSV file
csv_merged.to_csv('csv_merged.csv', index=False, header=False)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the 2nd column (index 1)
plt.plot(df[1])
plt.title('Plot of the 2nd Column')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```
#
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Conversion factor from Hartree to kcal/mol
hartree2kcm = 627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_MP2.csv', header=None)

# Extract x and z values
x = data.iloc[:, 0]
z = data.iloc[:, 1]

# Convert energy values
z = z - np.min(z)  # Subtract the minimum value from z
z = z * hartree2kcm  # Convert to kcal/mol

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, z, label='Energy')
ax.set_title('Energy Plot')
ax.set_xlabel("Values")
ax.set_ylabel("Energy [$\\Delta E$ in kcal/mol]")
ax.legend()

# Save and show the plot
plt.savefig('energy_plot.png')
plt.show()
```
#
```
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages

# Updated constant for unit conversion
hartree2kJmol = 2625.5  # Conversion from Hartree to kJ/mol

# Loop over all CSV files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        print(f"\nProcessing file: {filename}")
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract columns and rename for clarity
        var1 = data.iloc[:, 0].values
        var2 = data.iloc[:, 1].values
        energy = data.iloc[:, 2].values

        # Convert all energy values from Hartree to kJ/mol
        energy_kjmol = energy * hartree2kJmol

        # Normalize energy values in kJ/mol
        energy_kjmol = energy_kjmol - np.min(energy_kjmol)

        # Contour plot preparation
        var1_extended = np.append(var1, var2)
        var2_extended = np.append(var2, var1)
        energy_extended = np.append(energy_kjmol, energy_kjmol)

        # Create grid for interpolation
        xi = np.linspace(var1_extended.min(), var1_extended.max(), 1050)
        yi = np.linspace(var2_extended.min(), var2_extended.max(), 1050)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((var1_extended, var2_extended), energy_extended, (xi, yi), method='cubic')

        # Open a PDF file to save results
        pdf_filename = f"{filename.replace('.csv', '')}_results.pdf"
        with PdfPages(pdf_filename) as pdf:
            # Plot contour
            fig, ax = plt.subplots(figsize=(10, 8))
            levels = np.linspace(0, np.max(energy_kjmol), 50)
            cp = ax.contourf(xi, yi, zi, levels=levels, cmap='terrain', extend='both')
            plt.colorbar(cp)
            ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
            ax.set_xlabel("$r_1$ [$\AA$]")
            ax.set_ylabel("$r_2$ [$\AA$]")
            ax.set_title(f'Contour Plot for {filename}')
            pdf.savefig(fig)  # Save the figure to the PDF
            plt.close(fig)

            # Calculate minimum energy for cases where var1 == var2 and var1 != var2
            same_condition = (var1 == var2)
            diff_condition = (var1 != var2)

            # Save analysis results
            with open(pdf_filename.replace(".pdf", "_summary.txt"), 'w') as summary_file:
                if np.any(same_condition):
                    min_energy_same = np.min(energy_kjmol[same_condition])
                    min_index_same = np.where(same_condition & (energy_kjmol == min_energy_same))[0][0]
                    summary_file.write(
                        f"Minimum energy (var1 == var2): {min_energy_same:.4f} kJ/mol at (var1, var2) = ({var1[min_index_same]:.4f}, {var2[min_index_same]:.4f})\n"
                    )
                else:
                    min_energy_same = None
                    summary_file.write("No cases found where var1 == var2.\n")

                if np.any(diff_condition):
                    min_energy_diff = np.min(energy_kjmol[diff_condition])
                    min_index_diff = np.where(diff_condition & (energy_kjmol == min_energy_diff))[0][0]
                    var1_diff = var1[min_index_diff]
                    var2_diff = var2[min_index_diff]
                    diff_var1_var2 = abs(var1_diff - var2_diff)
                    summary_file.write(
                        f"Minimum energy (var1 != var2): {min_energy_diff:.4f} kJ/mol at (var1, var2) = ({var1_diff:.4f}, {var2_diff:.4f})\n"
                        f"Difference between var1 and var2 at min energy: {diff_var1_var2:.4f}\n"
                    )
                else:
                    min_energy_diff = None
                    diff_var1_var2 = None
                    summary_file.write("No cases found where var1 != var2.\n")

                # Compare absolute energy difference and apply the second condition
                if min_energy_same is not None and min_energy_diff is not None:
                    energy_difference = abs(min_energy_diff - min_energy_same)
                    summary_file.write(f"Energy difference: {energy_difference:.4f} kJ/mol\n")
                    if energy_difference >= 0.1 and (diff_var1_var2 is not None and diff_var1_var2 > 0.03):
                        summary_file.write("Result: distortion\n")
                    else:
                        summary_file.write("Result: no distortion\n")
                else:
                    summary_file.write("Comparison not possible due to missing data.\n")
```
#
```
import csv

# Read b.csv into a dictionary for easy lookup
b_data = {}
with open('b.csv', 'r') as b_file:
    b_reader = csv.reader(b_file)
    for row in b_reader:
        if row:
            b_data[row[0].strip()] = row[1:4]  # Store the 2nd, 3rd, and 4th columns for each molecule name

# Process a.csv and merge the data from b.csv
merged_rows = []
with open('a.csv', 'r') as a_file:
    a_reader = csv.reader(a_file)
    for row in a_reader:
        if row:
            mol_name = row[3].strip()  # Get the molecule name from the 4th column
            b_values = b_data.get(mol_name, ["", "", ""])  # Get corresponding values or empty if not found
            merged_rows.append(row + b_values)  # Add the values as the 7th, 8th, and 9th columns

# Write the merged data to a new CSV file
with open('merged.csv', 'w', newline='') as merged_file:
    merged_writer = csv.writer(merged_file)
    merged_writer.writerows(merged_rows)

print("Merging completed! Check the merged.csv file.")
```
#
```
import csv

# Read b.csv into a dictionary for easy lookup, skipping the header
b_data = {}
with open('b.csv', 'r') as b_file:
    b_reader = csv.reader(b_file)
    next(b_reader)  # Skip the header
    for row in b_reader:
        if row:
            b_data[row[0].strip()] = row[1:4]  # Store the 2nd, 3rd, and 4th columns for each molecule name

# Process a.csv, skipping the header, and merge the data from b.csv
merged_rows = []
with open('a.csv', 'r') as a_file:
    a_reader = csv.reader(a_file)
    header = next(a_reader)  # Read the header
    merged_header = header + ["Column7", "Column8", "Column9"]  # Add new column headers
    merged_rows.append(merged_header)

    for row in a_reader:
        if row:
            mol_name = row[3].strip()  # Get the molecule name from the 4th column
            b_values = b_data.get(mol_name, ["", "", ""])  # Get corresponding values or empty if not found
            merged_rows.append(row + b_values)  # Add the values as the 7th, 8th, and 9th columns

# Write the merged data to a new CSV file
with open('merged.csv', 'w', newline='') as merged_file:
    merged_writer = csv.writer(merged_file)
    merged_writer.writerows(merged_rows)

print("Merging completed! Check the merged.csv file.")
```
#
```import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Conversion factor from Hartree to kcal/mol
hartree2kcm = 627.509

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('contour_data_MP2.csv', header=None)

# Extract x and z values
x = data.iloc[:, 0]
z = data.iloc[:, 1]

# Convert energy values
z = z - np.min(z)  # Subtract the minimum value from z
z = z * hartree2kcm  # Convert to kcal/mol

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, z, label='Energy')
ax.set_title('Energy Plot')
ax.set_xlabel("Values")
ax.set_ylabel("Energy [$\\Delta E$ in kcal/mol]")
ax.legend()

# Save and show the plot
plt.savefig('energy_plot.png')
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the 2nd column (index 1)
plt.plot(df[1])
plt.title('Plot of the 2nd Column')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```
#
```
import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = '/home/atreyee/project/Project_AP1XY/ADC2_CC2_data'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read all CSV files into a list of DataFrames
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Attempt to convert the first column to numeric, replacing non-numeric values with NaN
merged_df[merged_df.columns[0]] = pd.to_numeric(merged_df[merged_df.columns[0]], errors='coerce')

# Sort the merged DataFrame based on the first column
merged_df = merged_df.sort_values(by=merged_df.columns[0])

# Save the merged and sorted DataFrame to a new CSV file
output_file = os.path.join(folder_path, 'merged_sorted.csv')
merged_df.to_csv(output_file, index=False)

print("CSV files merged and sorted successfully.")
```
#
```
import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = '/home/atreyee/project/Project_AP1XY/ADC2_CC2_data'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read all CSV files into a list of DataFrames
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Convert the first column to a uniform data type (e.g., string or numeric)
# Here, we try to convert to numeric and fall back to string if necessary
merged_df[merged_df.columns[0]] = pd.to_numeric(merged_df[merged_df.columns[0]], errors='coerce')

# Sort the merged DataFrame based on the first column
merged_df = merged_df.sort_values(by=merged_df.columns[0])

# Save the merged and sorted DataFrame to a new CSV file
output_file = os.path.join(folder_path, 'merged_sorted.csv')
merged_df.to_csv(output_file, index=False)

print("CSV files merged and sorted successfully.")
```
#
```
import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = '/home/atreyee/project/Project_AP1XY/ADC2_CC2_data'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read all CSV files into a list of DataFrames
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Convert the first column to a uniform data type (e.g., string or numeric)
# Here, we try to convert to numeric and fall back to string if necessary
merged_df[merged_df.columns[0]] = pd.to_numeric(merged_df[merged_df.columns[0]], errors='coerce')

# Sort the merged DataFrame based on the first column
merged_df = merged_df.sort_values(by=merged_df.columns[0])

# Save the merged and sorted DataFrame to a new CSV file
output_file = os.path.join(folder_path, 'merged_sorted.csv')
merged_df.to_csv(output_file, index=False)

print("CSV files merged and sorted successfully.")
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/mnt/data/gaps.csv'  # Adjust to your file path
data = pd.read_csv(file_path)

# Extract necessary columns
gap_values = data.iloc[:, 0]  # First column for gap values
molecule_names = data.iloc[:, 3].str.replace('_', ',')  # Fourth column, replace underscores with commas

# Define the x positions based on the sets
total_rows = len(gap_values)
x_positions = []
group_counts = [1, 4, 4, 4, 1, 1]  # Adjust the groupings as needed

if sum(group_counts) != total_rows:
    raise ValueError("The sum of group_counts must equal the number of rows in the data.")

x = 1  # Starting x position
for count in group_counts:
    x_positions.extend([x] * count)
    x += 1

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for better aesthetics
ax.set_xlim(0.5, max(x_positions) + 0.5)
ax.set_ylim(-0.35, 0.1)  # Adjust y-axis range for more space
ax.set_ylabel('Gap Value (eV)', fontsize=12)
ax.set_xlabel('Sets of Molecules', fontsize=12)
ax.set_title('Energy Gaps of Molecules', fontsize=14, fontweight='bold')

# Customize the plot's appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Function to adjust label position to avoid overlap
def adjust_label_position(index, gap_values, label_offset=0):
    # Check the distance between the current line and the previous one
    if index > 0:
        prev_gap = gap_values[index - 1]
        current_gap = gap_values[index]
        if abs(current_gap - prev_gap) < 0.05:  # If the lines are too close (adjust threshold as needed)
            return label_offset + 0.03  # Shift the label further
    return label_offset

# Plot the lines for each gap value at the corresponding x position
for i in range(len(gap_values)):
    # Draw a horizontal line for each gap value
    ax.hlines(y=gap_values[i], xmin=x_positions[i] - 0.1, xmax=x_positions[i] + 0.1, color='black', linewidth=1)
    
    # Adjust label position to avoid overlap
    label_offset = adjust_label_position(i, gap_values)

    # Annotate the gap value on the left side of the line
    ax.text(x_positions[i] - 0.15, gap_values[i] + label_offset, f'{gap_values[i]}', ha='right', fontsize=9, color='darkred')
    
    # Annotate the molecule name on the right side of the line
    ax.text(x_positions[i] + 0.15, gap_values[i] + label_offset, molecule_names[i], ha='left', fontsize=9, color='darkblue')

# Improve the overall aesthetics
plt.tight_layout()
plt.show()
```
# 
```
import pandas as pd
import matplotlib.pyplot as plt

# Define the energies and custom labels
data = {
    'Label': ['a', '2', '3', '4', '6', '8'],
    'Energy': [
        -385.691580683,
        -770.860688368 / 2,
        -771.336917778 / 2,
        -771.380824354 / 2,
        -771.382283738 / 2,
        -771.382816488 / 2
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the line passing through the first point
plt.figure(figsize=(10, 6))
plt.plot([1, len(df)], [df['Energy'][0], df['Energy'][0]], color='r', label='Line through Point a')

# Plot the curve joining the other points
plt.plot(range(2, len(df) + 1), df['Energy'][1:], marker='o', linestyle='-', color='b', label='Curve through other points')

# Add labels to each point
for i, row in df.iterrows():
    plt.text(i + 1, row['Energy'], f'{row["Label"]} ({row["Energy"]:.6f})', fontsize=10, ha='right')

# Add title and labels
plt.title('Energy Plot')
plt.xlabel('Index')
plt.ylabel('Energy')
plt.legend()

# Save the figure as a PDF
plt.savefig('energies_plot.pdf')

# Show the plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files without headers
csv1 = 'tda_results.csv'
csv2 = 'tddft_results.csv'

df1 = pd.read_csv(csv1, header=None)
df2 = pd.read_csv(csv2, header=None)

# Filter the rows based on the given ranges
filtered_df1 = df1[(df1.iloc[:, 8] >= 1.5) & (df1.iloc[:, 8] <= 4.0)]
filtered_df2 = df2[(df2.iloc[:, 8] >= 0.0) & (df2.iloc[:, 8] <= 2.0)]

# Find common indices in both filtered DataFrames
common_indices = filtered_df1.index.intersection(filtered_df2.index)

# Filter the DataFrames again to only include common indices
filtered_df1 = filtered_df1.loc[common_indices]
filtered_df2 = filtered_df2.loc[common_indices]

# Plot a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(filtered_df1.iloc[:, 8], filtered_df2.iloc[:, 8], c='blue', label='Filtered Data')
plt.xlabel('tda')
plt.ylabel('tddft')
plt.title('Scatter Plot')
plt.legend()
plt.grid(True)
plt.axis('square')
plt.show()

# Prepare the data for output
tda_column1 = filtered_df1.iloc[:, 0].reset_index(drop=True)
tda_column2 = filtered_df1.iloc[:, 1].reset_index(drop=True)
tda_column8 = filtered_df1.iloc[:, 8].astype(str).reset_index(drop=True)
tddft_column8 = filtered_df2.iloc[:, 8].astype(str).reset_index(drop=True)

# Combine columns with underscore
combined_column8 = tda_column8 + '_' + tddft_column8

# Create DataFrame for output
combined_data = pd.DataFrame({
    'Column1': tda_column1,
    'Column2': tda_column2,
    'CombinedColumn8': combined_column8
})

# Save to .smi file
combined_data.to_csv('combined_results.smi', sep='\t', index=False, header=False)

print("Data has been saved to 'combined_results.smi'.")
```
#
```
import os
import re
import csv

# Path to the parent directory where the folders are located
parent_dir = '/path/to/parent/directory'  # Modify this path accordingly

# Output CSV file
output_csv = 'fod_values.csv'

# List to hold the folder names and N_FOD values
data = []

# Regular expression to search for N_FOD values in fod.out files
fod_pattern = re.compile(r'N_FOD\s*=\s*(\d+\.\d+|\.\d+|\d+\.\d+E[+-]?\d+)')

# Walk through the parent directory
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    
    # Only consider directories (folders) that start with a number
    if os.path.isdir(folder_path) and folder_name[0].isdigit():
        fod_file_path = os.path.join(folder_path, 'fod.out')
        
        # Check if fod.out exists in the folder
        if os.path.exists(fod_file_path):
            with open(fod_file_path, 'r') as fod_file:
                # Read the file and search for the N_FOD value
                content = fod_file.read()
                match = fod_pattern.search(content)
                
                # If N_FOD is found, add the folder name and N_FOD value to data
                if match:
                    n_fod_value = match.group(1)
                    data.append([folder_name, n_fod_value])

# Sort the data by folder names (convert folder names to integers for proper numerical sorting)
data.sort(key=lambda x: int(x[0]))

# Write the results to a CSV file
with open(output_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Folder Name', 'N_FOD Value'])  # Write header
    writer.writerows(data)

print(f"CSV file '{output_csv}' has been written with the folder names and N_FOD values.")
```
#
```
import os
import pandas as pd

def merge_csv_side_by_side(base_directory, output_file):
    merged_df = None
    first_row = None
    folder_names = []

    # Walk through the base directory to find CSV files
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".csv"):
                folder_name = os.path.basename(root)
                folder_names.append(folder_name)

                file_path = os.path.join(root, file)

                # Read the CSV file
                df = pd.read_csv(file_path, header=None)

                if first_row is None:
                    first_row = df.iloc[0]  # Get the first row (common for all files)
                    merged_df = pd.DataFrame({ "Common": df.iloc[:, 0] })  # Initialize with the first column

                # Add the second column of the current file as a new column
                merged_df[folder_name] = df.iloc[:, 1].values

    # Ensure merged_df exists before modifying its headers
    if merged_df is not None:
        # Write the merged dataframe to a CSV file
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to {output_file}")
    else:
        print("No CSV files found in the directory.")

# Specify the base directory containing the folders and the output file name
base_directory = "path_to_your_base_directory"
output_file = "merged_output.csv"

merge_csv_side_by_side(base_directory, output_file)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'your_file.csv'

# Read the CSV file without a header
df = pd.read_csv(file_path, header=None)

# Plot the specified columns with respect to the 1st column
plt.plot(df[0], df[1], label='Column 2', color='blue')
plt.plot(df[0], df[3], label='Column 4', color='green')
plt.plot(df[0], df[5], label='Column 6', color='red')

plt.title('Plot of Columns 2, 4, 6 vs 1st Column')
plt.xlabel('1st Column')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
```
#
```
import os
import csv

# Input CSV file and target folder
input_csv = "negative_values.csv"
adc2_folder = "adc2"

# Create the adc2 folder if it doesn't exist
os.makedirs(adc2_folder, exist_ok=True)

# Template for the ADC(2) input file
adc2_template = """$molecule
  0  1
{coordinates}$end

$rem
jobtype             sp
method              adc(2)
basis               cc-pVDZ
aux_basis           rimp2-cc-pVDZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
"""

# Read the CSV and process each folder
with open(input_csv, "r") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header

    for row in reader:
        folder_name = row[0]  # First column is the folder name
        geom_file = os.path.join(folder_name, "geom_DFT_S0.xyz")

        # Check if geom_DFT_S0.xyz exists
        if os.path.isfile(geom_file):
            # Read the coordinates from geom_DFT_S0.xyz
            coordinates = ""
            with open(geom_file, "r") as geom:
                lines = geom.readlines()[2:]  # Skip the first two lines (header in XYZ files)
                for line in lines:
                    coordinates += line

            # Generate the ADC(2) input content
            adc2_input = adc2_template.format(coordinates=coordinates)

            # Create the subfolder in adc2
            target_subfolder = os.path.join(adc2_folder, folder_name)
            os.makedirs(target_subfolder, exist_ok=True)

            # Write the input file as all.com
            input_file = os.path.join(target_subfolder, "all.com")
            with open(input_file, "w") as outfile:
                outfile.write(adc2_input)

            print(f"Created ADC(2) input file: {input_file}")
        else:
            print(f"Geometry file not found: {geom_file}")

print("Process completed.")
```
#
```
#!/bin/bash

# Output CSV file
output_csv="results.csv"
echo "Folder,S1S0,fosc,T1S0,T2S0,S1T1,tT1S1,tT1T2" > $output_csv

# Loop through all directories
for dir in */; do
    # Check if tddft.out.bz2 exists in the folder
    file="$dir/tddft.out.bz2"
    if [ -f "$file" ]; then
        # Extract values
        S1S0=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   0' | sort -k6 -n | awk '{print $6}' | head -1)
        ind=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   0' | sort -k6 -n | awk '{print $2}' | head -1)
        ind=${ind/:/}
        fosc=$(bzgrep -A10 '         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' "$file" | grep "  $ind  " | awk '{print $4}')

        T1S0=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   2' | sort -k6 -n | awk '{print $6}' | head -1)
        T2S0=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   2' | sort -k6 -n | awk '{print $6}' | head -2 | tail -1)
        S1T1=$(echo "$S1S0 $T1S0" | awk '{print $1-$2}')
        tT1S1=$(echo "$T1S0 $S1S0" | awk '{print 2*$1-$2}')
        tT1T2=$(echo "$T1S0 $T2S0" | awk '{print 2*$1-$2}')

        # Append results to CSV
        echo "$dir,$S1S0,$fosc,$T1S0,$T2S0,$S1T1,$tT1S1,$tT1T2" >> $output_csv
    else
        echo "$dir,File not found,,,,,," >> $output_csv
    fi
done

# Inform the user
echo "Extraction completed. Results saved in $output_csv."
```
#
```
import os
import csv
import subprocess

def combine_xyz_files(folder_names, output_xyz="combined.xyz"):
    """Combine all geom_DFT_S0.xyz files into a single XYZ file."""
    with open(output_xyz, "w") as out_file:
        for folder in folder_names:
            xyz_path = os.path.join(folder, "geom_DFT_S0.xyz")
            if os.path.exists(xyz_path):
                with open(xyz_path, "r") as file:
                    out_file.writelines(file.readlines())

    print(f"Combined XYZ file written to {output_xyz}")

def convert_xyz_to_svg(input_xyz, output_svg="output.svg"):
    """Convert an XYZ file to SVG using the Open Babel CLI."""
    try:
        subprocess.run(["obabel", input_xyz, "-O", output_svg], check=True)
        print(f"SVG file created: {output_svg}")
    except FileNotFoundError:
        print("Error: Open Babel (obabel) is not installed or not in PATH.")

def main():
    # Path to the CSV file
    csv_file = "a.csv"

    # Read folder names from the first column of the CSV file
    folder_names = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            folder_names.append(row[0])

    # Combine XYZ files from the specified folders
    combined_xyz = "combined.xyz"
    combine_xyz_files(folder_names, combined_xyz)

    # Convert the combined XYZ file to SVG
    output_svg = "output.svg"
    convert_xyz_to_svg(combined_xyz, output_svg)

if __name__ == "__main__":
    main()
```
#
```
import os
import bz2

def extract_n_occ(tddft_file):
    """Extracts the N_occ value from tddft.out.bz2."""
    with bz2.open(tddft_file, 'rt') as f:
        for line in f:
            if 'N(Alpha)           :' in line:
                # Extract value and ignore decimal places
                n_occ = float(line.split(":")[1].strip())
                return int(n_occ)
    return None

def calculate_n_core(xyz_file):
    """Calculates N_core from geom_DFT_S0.xyz."""
    n_atoms = 0
    n_hydrogen = 0
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0].strip())  # First line is number of atoms
        for line in lines[2:]:  # Skip first two lines
            if 'H' in line.split()[0]:  # If atom is hydrogen
                n_hydrogen += 1
    n_core = n_atoms - n_hydrogen
    return n_core, n_hydrogen

def calculate_n_vir(n_core, n_hydrogen, n_electrons):
    """Calculates N_FROZEN_VIRTUAL (N_vir)."""
    n_vir = (14 * n_core + 5 * n_hydrogen) - (3 * n_electrons / 2 - 2 * n_core)
    return int(n_vir)

def create_adc2_input(folder, n_core, n_vir, xyz_file):
    """Creates the adc2 input file using geom_DFT_S0.xyz and calculated values."""
    with open(xyz_file, 'r') as f:
        coordinates = f.readlines()[2:]  # Skip the first two lines

    # Prepare the input file content
    input_content = f"""$molecule
  0  1
"""
    input_content += ''.join(coordinates)  # Add atom coordinates

    input_content += f"""
$end

$rem
jobtype             sp
method              adc(2)
N_FROZEN_CORE       {n_core}
N_FROZEN_VIRTUAL    {n_vir}
basis               cc-pVDZ
aux_basis           rimp2-cc-pVDZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
"""

    # Create the FC_FV folder structure
    fc_fv_folder = os.path.join('BNPAH', 'FC_FV', folder)
    os.makedirs(fc_fv_folder, exist_ok=True)

    # Write the content to all.com in the respective folder
    with open(os.path.join(fc_fv_folder, 'all.com'), 'w') as f:
        f.write(input_content)

def process_folders(base_folder):
    """Main function to process all subfolders within adc2 folder."""
    adc2_folder = os.path.join(base_folder, 'adc2')  # The 'adc2' folder
    if not os.path.isdir(adc2_folder):
        print("Error: 'adc2' folder not found.")
        return

    # List of subfolders in adc2
    subfolders = [f for f in os.listdir(adc2_folder) if os.path.isdir(os.path.join(adc2_folder, f))]

    # Iterate through subfolders in 'adc2' folder
    for folder in subfolders:
        folder_path = os.path.join(adc2_folder, folder)
        if os.path.isdir(folder_path):
            # Check if the same subfolder exists in BNPAH
            folder_in_bn = os.path.join(base_folder, folder)
            if os.path.isdir(folder_in_bn):
                # Find tddft.out.bz2 and geom_DFT_S0.xyz files in the subfolder in BNPAH
                tddft_file = os.path.join(folder_in_bn, 'tddft.out.bz2')
                xyz_file = os.path.join(folder_in_bn, 'geom_DFT_S0.xyz')

                if os.path.exists(tddft_file) and os.path.exists(xyz_file):
                    # Extract N_occ from tddft.out.bz2
                    n_occ = extract_n_occ(tddft_file)
                    if n_occ is None:
                        print(f"Error: N_occ not found in {tddft_file}. Skipping folder.")
                        continue

                    # Calculate N_core and N_hydrogen from geom_DFT_S0.xyz
                    n_core, n_hydrogen = calculate_n_core(xyz_file)

                    # Calculate N_electrons
                    n_electrons = n_occ - n_core

                    # Calculate N_FROZEN_VIRTUAL (N_vir)
                    n_vir = calculate_n_vir(n_core, n_hydrogen, n_electrons)

                    # Create the adc2 input file for this folder in the FC_FV folder
                    create_adc2_input(folder, n_core, n_vir, xyz_file)
                    print(f"Processed folder: {folder}")

if __name__ == "__main__":
    base_folder = 'BNPAH'  # The base folder containing 'adc2'
    process_folders(base_folder)
```
#
```
import os
import re

# Define the output file name
output_file = "output_table.txt"

# Initialize the output table
rows = []
rows.append(["FC", "FV", "S1", "T1", "STG"])

# Get the list of folders
folders = [f for f in os.listdir() if os.path.isdir(f) and f.startswith("FC_")]

for folder in folders:
    # Extract FC and FV values from the folder name
    match = re.match(r"FC_(\d+)_FV_(\d+)", folder)
    if not match:
        continue

    fc, fv = match.groups()

    # Construct the file path
    file_path = os.path.join(folder, "all1.out")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Extract S1 and T1 values from the file
        with open(file_path, "r") as f:
            content = f.readlines()

        S1 = None
        T1 = None

        for i, line in enumerate(content):
            if "(singlet" in line:
                for j in range(i + 1, i + 6):
                    if "Excitation" in content[j]:
                        S1 = float(content[j].split()[2])
                        break
            if "(triplet" in line:
                for j in range(i + 1, i + 6):
                    if "Excitation" in content[j]:
                        T1 = float(content[j].split()[2])
                        break

            if S1 is not None and T1 is not None:
                break

        # Calculate STG
        STG = S1 - T1 if S1 is not None and T1 is not None else None

        # Append the values to the rows list
        if S1 is not None and T1 is not None:
            rows.append([fc, fv, f"{S1:.3f}", f"{T1:.3f}", f"{STG:.3f}"])
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Write the rows to the output file
with open(output_file, "w") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")

print(f"Table written to {output_file}")
```
#
```
import os

# Total number of molecules to process
Nmol = 24
# File containing molecular coordinates
geomfile = "77.xyz"

# Get current working directory
filedir = os.getcwd()

# Open the input geometry file
with open(geomfile, 'r') as geom_file:
    for imol in range(1, Nmol + 1):
        # Read the number of atoms and title line
        line = geom_file.readline().strip()
        if line:
            Nat = int(line)  # Number of atoms
            title = geom_file.readline().strip()  # Title line containing molecule name
            print(Nat, title)

            # Create a subfolder named PAH_01, PAH_02, ..., PAH_24
            folder_name = f"PAH_{imol:02d}"
            os.mkdir(os.path.join(filedir, folder_name))

            # Write the molecule's data into a file in its subfolder
            output_file = os.path.join(filedir, folder_name, "geom_UFF.xyz")
            with open(output_file, "w") as inputfile:
                # Write header lines
                inputfile.write(f"{Nat}\n")
                inputfile.write(f"{title}\n")

                # Write atomic coordinates
                for iat in range(1, Nat + 1):
                    line = geom_file.readline().split()
                    sym = line[0]
                    R = [float(line[1]), float(line[2]), float(line[3])]
                    inputfile.write(f"{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n")
```
#
```
import os

# Total number of molecules to process
Nmol = 24
# File containing molecular coordinates
geomfile = "77.xyz"

# Get current working directory
filedir = os.getcwd()

# Open the input geometry file
with open(geomfile, 'r') as geom_file:
    for imol in range(1, Nmol + 1):
        # Read the number of atoms and title line
        line = geom_file.readline().strip()
        if line:
            Nat = int(line)  # Number of atoms
            title = geom_file.readline().strip()  # Title line containing molecule name
            print(Nat, title)

            # Create a subfolder named PAH_01, PAH_02, ..., PAH_24
            folder_name = f"PAH_{imol:02d}"
            os.mkdir(os.path.join(filedir, folder_name))

            # Write the molecule's data into a file in its subfolder
            output_file = os.path.join(filedir, folder_name, "geom_UFF.xyz")
            with open(output_file, "w") as inputfile:
                # Write header lines
                inputfile.write(f"{Nat}\n")
                inputfile.write(f"{title}\n")

                # Write atomic coordinates
                for iat in range(1, Nat + 1):
                    line = geom_file.readline().split()
                    sym = line[0]
                    R = [float(line[1]), float(line[2]), float(line[3])]
                    inputfile.write(f"{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n")
```
#
```
import os

# Total number of molecules to process
Nmol = 24
# File containing molecular coordinates
geomfile = "77.xyz"

# Get current working directory
filedir = os.getcwd()

# Open the input geometry file
with open(geomfile, 'r') as geom_file:
    for imol in range(1, Nmol + 1):
        # Read the number of atoms and title line
        line = geom_file.readline().strip()
        if line:
            Nat = int(line)  # Number of atoms
            title = geom_file.readline().strip()  # Title line containing molecule name
            print(Nat, title)

            # Create a subfolder named PAH_01, PAH_02, ..., PAH_24
            folder_name = f"PAH_{imol:02d}"
            os.mkdir(os.path.join(filedir, folder_name))

            # Write the molecule's data into a file in its subfolder
            output_file = os.path.join(filedir, folder_name, "geom_UFF.xyz")
            with open(output_file, "w") as inputfile:
                # Write header lines
                inputfile.write(f"{Nat}\n")
                inputfile.write(f"{title}\n")

                # Write atomic coordinates
                for iat in range(1, Nat + 1):
                    line = geom_file.readline().split()
                    sym = line[0]
                    R = [float(line[1]), float(line[2]), float(line[3])]
                    inputfile.write(f"{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n")
```
#
```
import os
import re

# Define the output file name
output_file = "output_table.txt"

# Initialize the output table
rows = []
rows.append(["FC", "FV", "S1", "T1", "STG"])

# Get the list of folders
folders = [f for f in os.listdir() if os.path.isdir(f) and f.startswith("FC_")]

for folder in folders:
    # Extract FC and FV values from the folder name
    match = re.match(r"FC_(\d+)_FV_(\d+)", folder)
    if not match:
        continue

    fc, fv = match.groups()

    # Construct the file path
    file_path = os.path.join(folder, "all1.out")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Extract S1 and T1 values from the file
        with open(file_path, "r") as f:
            content = f.readlines()

        S1 = None
        T1 = None

        for i, line in enumerate(content):
            if "(singlet" in line:
                for j in range(i + 1, i + 6):
                    if "Excitation" in content[j]:
                        S1 = float(content[j].split()[2])
                        break
            if "(triplet" in line:
                for j in range(i + 1, i + 6):
                    if "Excitation" in content[j]:
                        T1 = float(content[j].split()[2])
                        break

            if S1 is not None and T1 is not None:
                break

        # Calculate STG
        STG = S1 - T1 if S1 is not None and T1 is not None else None

        # Append the values to the rows list
        if S1 is not None and T1 is not None:
            rows.append([fc, fv, f"{S1:.3f}", f"{T1:.3f}", f"{STG:.3f}"])
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Write the rows to the output file
with open(output_file, "w") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")

print(f"Table written to {output_file}")
```
#
```
import os
import shutil  # Import shutil to copy files

# Total number of molecules to process
Nmol = 33059
# File containing molecular coordinates
geomfile = "77_pah.xyz"
# Template file to copy to each folder
template_file = "tddft.com"

# Get current working directory
filedir = os.getcwd()

# Open the input geometry file
with open(geomfile, 'r') as geom_file:
    for imol in range(1, Nmol + 1):
        # Read the number of atoms and title line
        line = geom_file.readline().strip()
        if line:
            Nat = int(line)  # Number of atoms
            title = geom_file.readline().strip()  # Title line containing molecule name
            print(Nat, title)

            # Create a subfolder named Mol_00001, Mol_00002, ..., Mol_33059
            folder_name = f"Mol_{imol:05d}"
            folder_path = os.path.join(filedir, folder_name)
            os.mkdir(folder_path)

            # Write the molecule's data into a file in its subfolder
            output_file = os.path.join(folder_path, "geom_UFF.xyz")
            with open(output_file, "w") as inputfile:
                # Write header lines
                inputfile.write(f"{Nat}\n")
                inputfile.write(f"{title}\n")

                # Write atomic coordinates
                for iat in range(1, Nat + 1):
                    line = geom_file.readline().split()
                    sym = line[0]
                    R = [float(line[1]), float(line[2]), float(line[3])]
                    inputfile.write(f"{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n")

            # Copy the template file to the folder
            shutil.copy(template_file, folder_path)
```
#
```
#!/bin/bash

# Ensure ext.sh is executed
bash ext.sh

# Summarize results
output_file="S1_T1_STG.txt"

echo "=== Jobs done ==="
tail -5 "$output_file"

echo "=== Small and Large STG ==="
sort -k4 -n "$output_file" | head -5
echo "---------------------------"
sort -k4 -n "$output_file" | tail -5

echo "=== Small and Large S1 ==="
sort -k2 -n "$output_file" | head -5
echo "---------------------------"
sort -k2 -n "$output_file" | tail -5

echo "=== Small and Large T1 ==="
sort -k3 -n "$output_file" | head -5
echo "---------------------------"
sort -k3 -n "$output_file" | tail -5
```
#
```
#!/bin/bash

# File to store the results
output_file="S1_T1_STG.txt"
> "$output_file"

# Loop through all folders
for dir in */; do
    file="${dir%/}/somefile.log"  # Adjust `somefile.log` to the actual file name in each folder
    if [[ -f "$file" ]]; then
        # Extract S1S0, T1S0, T2S0, and related values
        S1S0=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   0' | sort -k6 -n | awk '{print $6}' | head -1)
        ind=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   0' | sort -k6 -n | awk '{print $2}' | head -1)
        ind=${ind/:/}
        fosc=$(bzgrep -A10 '         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' "$file" | grep "  $ind  " | awk '{print $4}')
        
        T1S0=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   2' | sort -k6 -n | awk '{print $6}' | head -1)
        T2S0=$(bzgrep -A20 'STATE ' "$file" | grep '<S\*\*2> =   2' | sort -k6 -n | awk '{print $6}' | head -2 | tail -1)
        S1T1=$(echo "$S1S0 $T1S0" | awk '{print $1-$2}')
        tT1S1=$(echo "$T1S0 $S1S0" | awk '{print 2*$1-$2}')
        
        # Append the results to the output file
        echo "$dir $S1S0 $T1S0 $T2S0 $S1T1 $tT1S1 $fosc" >> "$output_file"
    fi
done

# Post-processing
echo "=== Jobs done ==="
tail -5 "$output_file"

echo "=== Small and Large STG ==="
sort -k4 -n "$output_file" | head -5
echo "---------------------------"
sort -k4 -n "$output_file" | tail -5

echo "=== Small and Large S1 ==="
sort -k2 -n "$output_file" | head -5
echo "---------------------------"
sort -k2 -n "$output_file" | tail -5

echo "=== Small and Large T1 ==="
sort -k3 -n "$output_file" | head -5
echo "---------------------------"
sort -k3 -n "$output_file" | tail -5
```
#
```
import os
import shutil  # Import shutil to copy files

# Total number of molecules to process
Nmol = 33059
# File containing molecular coordinates
geomfile = "77_pah.xyz"
# Template file to copy to each folder
template_file = "tddft.com"

# Get current working directory
filedir = os.getcwd()

# Open the input geometry file
with open(geomfile, 'r') as geom_file:
    for imol in range(1, Nmol + 1):
        # Read the number of atoms and title line
        line = geom_file.readline().strip()
        if line:
            Nat = int(line)  # Number of atoms
            title = geom_file.readline().strip()  # Title line containing molecule name
            print(Nat, title)

            # Create a subfolder named Mol_00001, Mol_00002, ..., Mol_33059
            folder_name = f"Mol_{imol:05d}"
            folder_path = os.path.join(filedir, folder_name)
            os.mkdir(folder_path)

            # Write the molecule's data into a file in its subfolder
            output_file = os.path.join(folder_path, "geom_UFF.xyz")
            with open(output_file, "w") as inputfile:
                # Write header lines
                inputfile.write(f"{Nat}\n")
                inputfile.write(f"{title}\n")

                # Write atomic coordinates
                for iat in range(1, Nat + 1):
                    line = geom_file.readline().split()
                    sym = line[0]
                    R = [float(line[1]), float(line[2]), float(line[3])]
                    inputfile.write(f"{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n")

            # Copy the template file to the folder
            shutil.copy(template_file, folder_path)
```
#
```
import os
import random
import numpy as np
import subprocess

def createXYZ(molecule, filename="PAH.xyz"):
    """Generate an XYZ file for the given molecule."""
    atom_map = {0: "C", 1: "N", -1: "B"}
    with open(filename, "w") as xyz_file:
        xyz_file.write(f"{len(molecule)}\n")
        xyz_file.write("Generated by GA\n")
        for i, atom_type in enumerate(molecule):
            atom = atom_map[atom_type]
            xyz_file.write(f"{atom} {i*1.5:.6f} 0.000000 0.000000\n")

def runtddft():
    """Run ORCA TD-DFT calculation."""
    os.system("bash runorca.sh")

def findSTG(output_file="tddft.out"):
    """Extract the singlet-triplet gap from ORCA output."""
    try:
        # Run the equivalent of the bzgrep command to find the singlet and triplet states
        bzgrep_command = f"bzgrep -A20 'STATE ' {output_file} | grep '<S**2> =   0' | sort -k6 -n | awk '{{print $6}}' | head -1"
        S1S0 = subprocess.check_output(bzgrep_command, shell=True, text=True).strip()
        
        bzgrep_command_ind = f"bzgrep -A20 'STATE ' {output_file} | grep '<S**2> =   0' | sort -k6 -n | awk '{{print $2}}' | head -1"
        ind = subprocess.check_output(bzgrep_command_ind, shell=True, text=True).strip().replace(":", "")
        
        bzgrep_command_fosc = f"bzgrep -A10 '         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' {output_file} | grep '  {ind}  ' | awk '{{print $4}}'"
        fosc = subprocess.check_output(bzgrep_command_fosc, shell=True, text=True).strip()
        
        bzgrep_command_T1S0 = f"bzgrep -A20 'STATE ' {output_file} | grep '<S**2> =   2' | sort -k6 -n | awk '{{print $6}}' | head -1"
        T1S0 = subprocess.check_output(bzgrep_command_T1S0, shell=True, text=True).strip()
        
        bzgrep_command_T2S0 = f"bzgrep -A20 'STATE ' {output_file} | grep '<S**2> =   2' | sort -k6 -n | awk '{{print $6}}' | head -2 | tail -1"
        T2S0 = subprocess.check_output(bzgrep_command_T2S0, shell=True, text=True).strip()
        
        # Debug: print the extracted values
        print(f"S1S0: {S1S0}, T1S0: {T1S0}, fosc: {fosc}, ind: {ind}")

        # Check if the values are empty before calculating the gap
        if not S1S0 or not T1S0:
            print("Error: Empty S1S0 or T1S0 values")
            return float("inf")
        
        # Calculate the singlet-triplet gap
        S1T1_gap = float(S1S0) - float(T1S0)
        return S1T1_gap
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting S1-T1 gap: {e}")
    return float("inf")

def fitness_function_otf(molecule):
    """Calculate the fitness of a molecule based on the S1-T1 gap."""
    count_C = molecule.count(0)
    count_N = molecule.count(1)
    count_B = molecule.count(-1)

    if count_B != count_N:
        return float("inf")  # Penalize invalid structures

    createXYZ(molecule)
    runtddft()
    gap = findSTG()

    return gap

def initialize_population(size, num_sites):
    """Initialize a population of random molecules."""
    return [[random.choice([0, -1, 1]) for _ in range(num_sites)] for _ in range(size)]

def crossover(parent1, parent2):
    """Perform single-point crossover between two parents."""
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(molecule, mutation_rate=0.1):
    """Mutate a molecule with a given mutation rate."""
    return [random.choice([0, -1, 1]) if random.random() < mutation_rate else gene for gene in molecule]

def genetic_algorithm(population_size=20, generations=50, num_sites=10):
    """Main Genetic Algorithm loop."""
    seen_best_molecules = set()
    population = initialize_population(population_size, num_sites)

    for gen in range(generations):
        fitness_scores = [fitness_function_otf(molecule) for molecule in population]

        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        best_molecule = population[0]
        best_gap = fitness_scores[sorted_indices[0]]

        if tuple(best_molecule) not in seen_best_molecules:
            seen_best_molecules.add(tuple(best_molecule))
            count_C = best_molecule.count(0)
            count_N = best_molecule.count(1)
            count_B = best_molecule.count(-1)
            print(f"Generation {gen+1}: Lowest S1-T1 gap = {best_gap:.4f}, Molecule = {best_molecule}, C{count_C}_B{count_B}_N{count_N}")

        next_population = population[:population_size // 2]
        while len(next_population) < population_size:
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            next_population.append(child)

        population = next_population

    return best_molecule, best_gap

# Main execution
if __name__ == "__main__":
    best_molecule, best_gap = genetic_algorithm(population_size=20, generations=100, num_sites=10)
    print("Genetic Algorithm Optimized Molecule:", best_molecule)
    print("S1-T1 Gap is:", best_gap)
```
#
```
import csv

# Assuming molecule_cache, symmetry_maps, apply_symmetry, findSTG, runtddft, createXYZ, and other functions/variables are defined elsewhere

molecule_str = "".join(map(str, molecule))
if molecule_str in molecule_cache:
    S1, T1, gap = molecule_cache[molecule_str]
else:
    # Read "molecules_gap.csv" to check if molecule is already there
    molecule_found = False
    with open("molecules_gap.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == molecule_str:
                S1, T1, gap = float(row[1]), float(row[2]), float(row[3])
                molecule_found = True
                break

    if not molecule_found:
        # If molecule is not found, create XYZ and compute the values
        createXYZ(molecule)
        runtddft()
        S1, T1, gap = findSTG()

        molecule_cache[molecule_str] = (S1, T1, gap)

        if S1 < 0 or T1 < 0:
            gap = float("inf")

        # Write the data to "molecules_gap.csv"
        with open("molecules_gap.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([molecule_str, S1, T1, gap, f"C{count_C}_B{count_B}_N{count_N}"])

            # Apply symmetry maps and append to the file
            for symmetry_map in symmetry_maps:
                transformed_molecule = tuple(apply_symmetry(molecule, symmetry_map))
                transformed_molecule_str = "".join(map(str, transformed_molecule))
                writer.writerow([transformed_molecule_str, S1, T1, gap, f"C{count_C}_B{count_B}_N{count_N}"])

# Print and write results to molecules_gap.txt
print(molecule, S1, T1, gap, f"C{count_C}_B{count_B}_N{count_N}")
with open("molecules_gap.txt", "a") as results_file:
    results_file.write(f"{molecule} {S1} {T1} {gap} C{count_C}_B{count_B}_N{count_N}\n")
```
#
```
def fitness_function_otf(molecule):
    molecule_str = "".join(map(str, molecule))
    result = check_molecule_in_csv(molecule_str)
    
    if result is None:
        print(f"Molecule {molecule_str} not found in CSV.")
        S1, T1, gap = float('Inf'), float('Inf'), float('Inf')  # Default values for missing molecules
    else:
        S1, T1, gap = result
        if float(S1) < 0 or float(T1) < 0:
            gap = float('Inf')  # Assign a high fitness value for invalid cases
    
    # Count the occurrences of each atom type
    count_C = molecule.count(0)
    count_N = molecule.count(1)
    count_B = molecule.count(-1)

    # Write results to file
    with open("molecules_gap.txt", "a") as results_file:
        results_file.write(f"{molecule} {S1} {T1} {gap} C{count_C}_B{count_B}_N{count_N}\n")

    return gap
```
#
```
import csv
import os

# Cache for previously computed molecules
molecule_cache = {}

def createXYZ(molecule, filename="PAH.xyz"):
    # Function to create an XYZ file based on the molecule
    with open('./PAH_store.xyz', 'r') as file:
        lines = file.readlines()

    atom_map = {0: "C", 1: "N", -1: "B"}

    with open(filename, "w") as xyz_file:
        for iline, line in enumerate(lines):
            if iline == 0:
                xyz_file.write(line)
            elif iline == 1:
                xyz_file.write("Generated by GA\n")
            else:
                xyzline = line.split()
                atom = xyzline[0]
                if iline < len(molecule) + 2:
                    atom = atom_map[molecule[iline - 2]]
                x = float(xyzline[1])
                y = float(xyzline[2])
                z = float(xyzline[3])
                xyz_file.write(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")

def findSTG():
    # Placeholder function for extracting S1, T1, and gap (STG)
    # Implement your TDDFT extraction logic here
    S1 = 1.2  # Placeholder value
    T1 = 0.8  # Placeholder value
    gap = S1 - T1  # Example gap calculation (S1 - T1)
    return S1, T1, gap

def apply_symmetry(molecule, symmetry_map):
    # Function to apply symmetry transformation (to be defined based on your symmetry operations)
    # This is a placeholder, modify it according to your symmetry logic
    return molecule

def run_opt(molecule):
    # Function for running ORCA optimization
    print(f"Running ORCA optimization for molecule: {molecule}")
    createXYZ(molecule, "opt.com")
    os.system("bash runorca1.sh")  # Run ORCA optimization script
    print(f"ORCA optimization completed for molecule: {molecule}")

def run_adc2(molecule):
    # Function for running ADC2 calculation
    print(f"Running ADC2 calculation for molecule: {molecule}")
    # Implement ADC2 calculation here
    os.system("bash runqchem.sh")  # Run ADC2 calculation
    print(f"ADC2 calculation completed for molecule: {molecule}")

def fitness_function_otf(molecule):
    # Assuming you are working with molecules represented as a list of atom types (e.g., C, N, B)
    count_C = molecule.count(0)
    count_N = molecule.count(1)
    count_B = molecule.count(-1)

    # If the number of B and N atoms are not equal, we do not perform TDDFT or any other calculations
    if count_B != count_N:
        S1 = float("inf")
        T1 = float("inf")
        gap = float("inf")
    else:
        molecule_str = "".join(map(str, molecule))
        if molecule_str in molecule_cache:
            S1, T1, gap = molecule_cache[molecule_str]
        else:
            # Generate XYZ and run TDDFT (Placeholder for actual TDDFT logic)
            createXYZ(molecule)
            runtddft()  # Placeholder for your TDDFT function call
            S1, T1, gap = findSTG()

            molecule_cache[molecule_str] = (S1, T1, gap)

            if S1 < 0 or T1 < 0:
                gap = float("inf")

            with open("molecules_gap.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([molecule_str, S1, T1, gap, f"C{count_C}_B{count_B}_N{count_N}"])
                # Apply each symmetry map and append to data
                for symmetry_map in symmetry_maps:
                    transformed_molecule = tuple(apply_symmetry(molecule, symmetry_map))
                    transformed_molecule_str = "".join(map(str, transformed_molecule))
                    writer.writerow([transformed_molecule_str, S1, T1, gap, f"C{count_C}_B{count_B}_N{count_N}"])

    print(molecule, S1, T1, gap, f"C{count_C}_B{count_B}_N{count_N}")
    with open("molecules_gap.txt", "a") as results_file:
        results_file.write(f"{molecule} {S1} {T1} {gap} C{count_C}_B{count_B}_N{count_N}\n")

    # Check the condition (S1 > 0, T1 > 0, STG < 0) and call ADC2 and optimization functions
    if S1 > 0 and T1 > 0 and gap < 0:
        run_opt(molecule)  # Run ORCA optimization
        run_adc2(molecule)  # Run ADC2 calculation

    return gap

# Example usage of the function for different molecules
molecules = [
    [0, 1, 0, -1],  # Example molecule, change it to the actual data
    [0, 0, 1, -1],
    [1, 1, 0, -1]
]

for molecule in molecules:
    fitness_function_otf(molecule)  # Process each molecule
```
#
```
from rdkit import Chem

# Input and output file names
input_file = "input.smi"
valid_output_file = "valid_smiles.smi"
invalid_output_file = "invalid_smiles.txt"

# Open the input file and prepare output files
with open(input_file, "r") as infile, \
     open(valid_output_file, "w") as valid_outfile, \
     open(invalid_output_file, "w") as invalid_outfile:
    
    for line in infile:
        # Remove extra spaces and split by whitespace
        smiles = line.strip()
        
        # Skip empty lines
        if not smiles:
            continue
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Write valid SMILES to the valid output file
            valid_outfile.write(f"{Chem.MolToSmiles(mol)}\n")
        else:
            # Write invalid SMILES to the invalid output file
            invalid_outfile.write(f"Invalid: {smiles}\n")

print("Validation completed!")
print(f"Valid SMILES saved in: {valid_output_file}")
print(f"Invalid SMILES saved in: {invalid_output_file}")
```
#
```
import pandas as pd

def create_smi_for_first_23(csv_file, output_smi_file):
    """
    Create a .smi file for the first 23 molecules, with the first column as SMILES
    (second column of b.csv) and the second column as molecule names (first column of b.csv).

    Args:
        csv_file (str): Path to the CSV file.
        output_smi_file (str): Path to the output .smi file.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)

        # Get the first 23 rows of data
        first_23_data = data.head(23)

        # Create the .smi content: first column is SMILES, second column is molecule name
        smi_content = first_23_data[[data.columns[1], data.columns[0]]]

        # Write to the .smi file
        smi_content.to_csv(output_smi_file, header=False, index=False, sep='\t')

        print(f".smi file created successfully: {output_smi_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
csv_file = 'b.csv'  # Replace with the path to the second CSV file
output_smi_file = 'new.smi'  # Replace with the desired output .smi file path
create_smi_for_first_23(csv_file, output_smi_file)
```
#
```
import os
import shutil

# Paths to the main folders
source_main_folder = "SCS-PBE-QIDH_VDZ_33059"
destination_main_folder = "FOD_33059"
sample_fod_file = os.path.join(destination_main_folder, "fod.com")

# Check if the destination main folder exists, if not, create it
if not os.path.exists(destination_main_folder):
    os.makedirs(destination_main_folder)

# Iterate through all subfolders in the source folder
for root, dirs, files in os.walk(source_main_folder):
    for dir_name in dirs:
        source_subfolder = os.path.join(root, dir_name)
        destination_subfolder = os.path.join(destination_main_folder, dir_name)
        
        # Create the same subfolder structure in the destination folder
        if not os.path.exists(destination_subfolder):
            os.makedirs(destination_subfolder)
        
        # Copy geom_DFT_S0.xyz if it exists
        geom_file_path = os.path.join(source_subfolder, "geom_DFT_S0.xyz")
        if os.path.exists(geom_file_path):
            shutil.copy(geom_file_path, destination_subfolder)
        
        # Copy fod.com to the subfolder
        if os.path.exists(sample_fod_file):
            shutil.copy(sample_fod_file, destination_subfolder)

print("Files copied successfully!")
```
# 
```
import os
from rdkit import Chem
from rdkit.Chem import RDConfig
import sys

# Add SA_Score path
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Input and output files
input_file = "BNPAH_33059_fixedgeom_mod.smi"
output_file = "33059_sa_score.txt"

def calculate_sa_scores(input_file, output_file):
    """
    Calculate synthetic accessibility (SA) scores for molecules from a SMILES file.

    Args:
        input_file (str): Path to the input file containing SMILES strings.
        output_file (str): Path to the output file where SA scores will be written.

    Returns:
        None
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Write header to the output file
            outfile.write("Molecule_Name\tSMILES\tSA_Score\n")

            # Read SMILES strings and process each
            for idx, line in enumerate(infile, start=1):
                smiles = line.strip()

                if not smiles:  # Skip empty lines
                    continue

                molecule_name = f"Mol{idx:02d}"

                try:
                    # Convert SMILES to RDKit molecule
                    mol = Chem.MolFromSmiles(smiles)

                    if mol is not None:
                        # Calculate SA score
                        sa_score = sascorer.calculateScore(mol)
                        # Write to output file
                        outfile.write(f"{molecule_name}\t{smiles}\t{sa_score:.3f}\n")
                    else:
                        outfile.write(f"{molecule_name}\t{smiles}\tInvalid_SMILES\n")

                except Exception as e:
                    outfile.write(f"{molecule_name}\t{smiles}\tError: {str(e)}\n")

        print(f"SA scores successfully written to {output_file}.")

    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# Run the function
if __name__ == "__main__":
    calculate_sa_scores(input_file, output_file)
```
#
```
import
```
#
```
density plot
```
#
```
#!/bin/bash

output_file="S1_T1_STG.txt"
rm -f $output_file

for dir in Mol_*; do
    file="$dir/tddft.out"
    if [ -f "$file" ]; then
        S1S0=$(grep -A20 'STATE ' "$file" | grep '<S\*\*2> =   0' | sort -k6 -n | awk '{print $6}' | head -1)
        ind=$(grep -A20 'STATE ' "$file" | grep '<S\*\*2> =   0' | sort -k6 -n | awk '{print $2}' | head -1)
        ind=${ind/:/}
        fosc=$(grep '0-1A  ->  1-1A' "$file" | head -1 | awk '{print $7}')
        T1S0=$(grep -A20 'STATE ' "$file" | grep '<S\*\*2> =   2' | sort -k6 -n | awk '{print $6}' | head -1)
        S1T1=$(echo "$S1S0 $T1S0" | awk '{print $1-$2}')

        if (( $(echo "$S1S0 > 0" | bc -l) )) && (( $(echo "$T1S0 > 0" | bc -l) )); then
            printf "%-20s %8.3f %8.3f %8.3f %10.5f\n" "$dir" "$S1S0" "$T1S0" "$S1T1" "$fosc" >> $output_file
        fi
    fi
done

```
#
```
import os

# Paths and templates
top_30_names_file = "top_30_names.txt"
scs_folder = "SCS-PBE-QIDH_VDZ_33059"
output_base_folder = "top30_ADC2_VDZ"

adc2_template = """$molecule
  0  1
{coordinates}$end

$rem
jobtype             sp
method              adc(2)
basis               cc-pVDZ
aux_basis           rimp2-cc-pVDZ
mem_total           64000
mem_static          1000
maxscf              1000
cc_symmetry         false
ee_singlets         3
ee_triplets         3
sym_ignore          true
ADC_DAVIDSON_MAXITER 300
ADC_DAVIDSON_CONV 5
$end
"""

def read_folder_names(file_path):
    """Read folder names from the given file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def extract_coordinates(file_path):
    """Extract coordinates from the XYZ file, skipping the first two lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return ''.join(lines[2:])  # Skip the first two lines

def create_adc2_input(coordinates, output_path):
    """Create the ADC(2) input file."""
    input_content = adc2_template.format(coordinates=coordinates)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "all.com"), 'w') as file:
        file.write(input_content)

def process_molecules(top_30_names, scs_folder, output_base_folder):
    """Main process to create ADC(2) inputs for all top 30 molecules."""
    for idx, molecule_name in enumerate(top_30_names):
        scs_molecule_folder = os.path.join(scs_folder, molecule_name)
        geom_file = os.path.join(scs_molecule_folder, "geom_DFT_S0.xyz")
        
        if os.path.exists(geom_file):
            try:
                coordinates = extract_coordinates(geom_file)
                
                # Generate sequential folder names: Mol_00001, Mol_00002, ...
                output_folder_name = f"Mol_{idx+1:05d}"
                output_folder = os.path.join(output_base_folder, output_folder_name)
                
                create_adc2_input(coordinates, output_folder)
            except Exception as e:
                print(f"Error processing {molecule_name}: {e}")
        else:
            print(f"Warning: geom_DFT_S0.xyz not found for molecule {molecule_name} in {scs_folder}")

if __name__ == "__main__":
    # Step 1: Read the top 30 molecule names
    top_30_names = read_folder_names(top_30_names_file)
    
    # Step 2: Process each molecule and create respective inputs
    process_molecules(top_30_names, scs_folder, output_base_folder)
    
    print(f"ADC(2) inputs have been created in the folder: {output_base_folder}")
```
#
```
memory,8,g
charge=0

gdirect
symmetry,nosym;orient,noorient

geometry={
 N         -0.0000000000        0.0000000000        0.0000000000
 C          1.4025590960        0.0000000000        0.0000000000
 C         -0.7012795480       -1.2146518075        0.0000000000
 ...
}

basis={
default,vdz
set,mp2fit
default,vdz/mp2fit
set,jkfit
default,vdz/jkfit }

hf

{lt-df-lcc2                     !ground state CC2
eom,-6.1,triplet=1              !triplet states
eomprint,popul=-1,loceom=-1 }   !minimize the output same thing for cc2 inp
```
#
```
# Read the names from both files
with open('top_30_names.txt', 'r') as f:
    top_30_names = set(f.read().splitlines())

with open('top82.txt', 'r') as f:
    top82_names = f.read().splitlines()

# Find names in top82 that are not in top_30_names
names_not_in_top_30 = [name for name in top82_names if name not in top_30_names]

# Write the 52 names into a new file
with open('names_not_in_top_30.txt', 'w') as f:
    f.write("\n".join(names_not_in_top_30))

print("The file 'names_not_in_top_30.txt' has been created with 52 names.")
```
#
```
import os
import shutil

# Number of molecules
num_molecules = 82

# Open the top82.xyz file once to read all lines
with open('top82.xyz', 'r') as file:
    lines = file.readlines()

# Create folders named Mol_00001 to Mol_00082
for i in range(1, num_molecules + 1):
    folder_name = f'Mol_{i:05d}'  # Formatting the folder name as Mol_00001, Mol_00002, ...
    os.makedirs(folder_name, exist_ok=True)

    # Find the starting line for the current molecule
    start_line = (i - 1) * 2  # The first 2 lines in each section are the atom count and molecule name
    
    # The first line contains the atom count, the second line is the molecule name
    num_atoms = int(lines[start_line].strip())  # Get the number of atoms for the molecule

    # The coordinates for the current molecule span the next `num_atoms` lines
    molecule_lines = lines[start_line + 1:start_line + 1 + num_atoms]

    # Save coordinates in geom.xyz within the respective folder
    with open(f'{folder_name}/geom.xyz', 'w') as geom_file:
        # Write the atom count and molecule name as the first two lines in the geom.xyz
        geom_file.write(f"{num_atoms}\n")
        geom_file.write(f"Mol_{i:05d}\n")
        # Write the atom coordinates
        geom_file.writelines(molecule_lines)

    # Copy opt.com to the folder
    shutil.copy('opt.com', f'{folder_name}/opt.com')

print("Folders created and files copied successfully.")
```
#
```
import os
import shutil

# Paths for source and destination directories
source_dir = "top82_TPSSh_freq"
destination_dir = "top82_G16_wB97XD3_opt_freq"

# Gaussian input template
gaussian_template = """%mem=64GB
%nprocs=18
#P  wB97XD/cc-pVDZ  SCF(maxcycles=100,verytight)  Int(Grid=ultrafine) Opt(maxcyc=1000, calcall, verytight) Freq

Test

0 1
"""

def create_gaussian_input(source_path, dest_path):
    """Reads geom_DFT_S0.xyz and creates Gaussian input."""
    try:
        with open(source_path, "r") as xyz_file:
            lines = xyz_file.readlines()

        # Extract coordinates, skipping first 2 lines
        coordinates = lines[2:]

        # Write the Gaussian input file
        with open(dest_path, "w") as g_input:
            g_input.write(gaussian_template)
            g_input.writelines(coordinates)
            g_input.write("\n\n\n\n")  # Add 4 empty lines at the end

        print(f"Created: {dest_path}")

    except Exception as e:
        print(f"Error processing {source_path}: {e}")

# Iterate through folders and create Gaussian inputs
for root, dirs, files in os.walk(source_dir):
    for folder in dirs:
        source_folder = os.path.join(root, folder)
        xyz_file = os.path.join(source_folder, "geom_DFT_S0.xyz")

        if os.path.exists(xyz_file):
            # Create corresponding folder in destination directory
            relative_path = os.path.relpath(source_folder, source_dir)
            dest_folder = os.path.join(destination_dir, relative_path)
            os.makedirs(dest_folder, exist_ok=True)

            # Define the output Gaussian input file path
            gaussian_input_path = os.path.join(dest_folder, "input.com")

            # Create the Gaussian input file
            create_gaussian_input(xyz_file, gaussian_input_path)

print("All Gaussian input files have been created.")
```
#
```
import os

# Template for inp.com
inp_template = """memory,8,g
charge=0

gdirect
symmetry,nosym;orient,noorient

geometry={{
{coordinates}}}

basis={{
default,vdz
set,mp2fit
default,vdz/mp2fit
set,jkfit
default,vdz/jkfit }}

hf

{{lt-df-lcc2                     !ground state CC2
eom,-6.1,triplet=1              !triplet states
eomprint,popul=-1,loceom=-1 }}   !minimize the output same thing for cc2 inp
"""

def extract_coordinates(file_path):
    """Extract coordinates from the XYZ file, skipping the first two lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return ''.join(lines[2:])  # Skip the first two lines

def create_inp_file(coordinates, output_path):
    """Create the inp.com file."""
    inp_content = inp_template.format(coordinates=coordinates)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "inp.com"), 'w') as file:
        file.write(inp_content)

def process_xyz_file(source_base, target_base, folder_name):
    """Process a single folder and create inp.com if geom_DFT_S0.xyz exists."""
    source_folder = os.path.join(source_base, folder_name)
    geom_file = os.path.join(source_folder, "geom_DFT_S0.xyz")
    
    if os.path.exists(geom_file):
        try:
            coordinates = extract_coordinates(geom_file)
            output_folder = os.path.join(target_base, folder_name)
            create_inp_file(coordinates, output_folder)
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
    else:
        print(f"Warning: geom_DFT_S0.xyz not found in {source_folder}")

def process_all_folders(source_base, target_base, start_index=1, end_index=8):
    """Process folders named P1, P2, ..., P8."""
    for i in range(start_index, end_index + 1):
        folder_name = f"P{i}"  # Generate folder names like P1, P2, etc.
        print(f"Processing {folder_name}...")
        process_xyz_file(source_base, target_base, folder_name)

if __name__ == "__main__":
    # Define source and target base folders
    source_base = "wB97XD3_TZVP_opt_freq"
    target_base = "LCC2_VDZ"

    # Process folders from P1 to P8
    process_all_folders(source_base, target_base, start_index=1, end_index=8)

    print(f"inp.com files have been created in the folder: {target_base}")
```
#
```
import os

# Read the folder names from stable_37.txt
with open('stable_37.txt', 'r') as f:
    folder_names = [line.strip() for line in f]

# Open the combined output file
with open('stable_37_TPSSh.xyz', 'w') as combined_xyz:
    # Iterate over each folder
    for folder in folder_names:
        # Construct the path to the TPSSh.xyz file in the folder
        file_path = os.path.join(folder, 'TPSSh.xyz')
        
        # Check if the file exists in the folder
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                # Read the contents of TPSSh.xyz and append to the combined file
                combined_xyz.write(f.read())
                combined_xyz.write('\n')  # Add a newline to separate each file's content
        else:
            print(f"File {file_path} does not exist.")
```
#
```
import os
import shutil

# Read folder names from the stable_41.txt file
with open('stable_41.txt', 'r') as file:
    folder_names = [line.strip() for line in file if line.strip()]

# Ensure the target directory exists
fod_dir = 'FOD_41'
os.makedirs(fod_dir, exist_ok=True)

# Check for the presence of fod.com
fod_com_path = os.path.join(fod_dir, 'fod.com')
if not os.path.exists(fod_com_path):
    raise FileNotFoundError(f"The file 'fod.com' was not found in {fod_dir}.")

# Iterate through folder names and perform operations
for i, folder_name in enumerate(folder_names, start=1):
    # Create target subfolder in FOD_41
    target_folder_name = f"Mol_{i:05d}"
    target_folder_path = os.path.join(fod_dir, target_folder_name)
    os.makedirs(target_folder_path, exist_ok=True)

    # Define source and target paths for geom_DFT_S0.xyz
    source_xyz_path = os.path.join(folder_name, 'geom_DFT_S0.xyz')
    target_xyz_path = os.path.join(target_folder_path, 'geom_DFT_S0.xyz')

    if not os.path.exists(source_xyz_path):
        print(f"Warning: {source_xyz_path} does not exist. Skipping.")
        continue

    # Copy geom_DFT_S0.xyz to the target folder
    shutil.copy(source_xyz_path, target_xyz_path)

    # Copy fod.com to the target folder
    shutil.copy(fod_com_path, target_folder_path)

print("All files have been processed and copied.")
```
#
```
import os

# Function to calculate necessary parameters
def calculate_parameters(heavy_atoms, hydrogen_atoms):
    N_heavy = heavy_atoms
    N_Hydrogen = hydrogen_atoms

    # Number of electrons
    N_electrons = N_heavy * 6 + N_Hydrogen * 1

    # No. of 1s core MOs
    N_core = N_heavy

    # Basis functions for H in cc-pVDZ
    N_bas_VDZ_H = 5

    # Basis functions for B/C/N in cc-pVDZ
    N_bas_VDZ_CBN = 14

    # Molecular orbitals in cc-pVDZ
    N_MOs = 14 * N_heavy + 5 * N_Hydrogen

    # Occupied MOs
    N_occ = N_electrons // 2

    # Unoccupied/virtual MOs
    N_vir = N_MOs - N_occ

    N_FROZEN_CORE = N_core
    N_FROZEN_VIRTUAL = N_MOs - N_occ - 2 * (N_occ - N_core)

    return N_FROZEN_CORE, N_FROZEN_VIRTUAL

# Function to process geom_DFT_S0.xyz and create all1.com
def process_xyz_file(source_folder, target_folder, folder_name):
    xyz_file_path = os.path.join(source_folder, folder_name, "geom_DFT_S0.xyz")

    if os.path.isfile(xyz_file_path):
        # Read the .xyz file
        with open(xyz_file_path, 'r') as f:
            lines = f.readlines()

        # Extract atom data from the .xyz file
        atom_count = int(lines[0].strip())  # First line gives the number of atoms
        heavy_atoms = 0
        hydrogen_atoms = 0
        atom_data = []
        for line in lines[2:]:  # Skip the first two lines (atom count and comment)
            parts = line.split()
            atom_type = parts[0]
            x, y, z = parts[1:4]  # Coordinates
            if atom_type in ['B', 'C', 'N']:
                heavy_atoms += 1
            elif atom_type == 'H':
                hydrogen_atoms += 1
            atom_data.append(f"  {atom_type}   {x}   {y}   {z}\n")

        # Calculate parameters
        N_FROZEN_CORE, N_FROZEN_VIRTUAL = calculate_parameters(heavy_atoms, hydrogen_atoms)

        # Prepare new .com file content
        new_com_content = "$molecule\n"
        new_com_content += "  0  1\n"  # Only one occurrence of "0 1"
        new_com_content += "".join(atom_data)
        new_com_content += "$end\n\n"
        new_com_content += "$rem\n"
        new_com_content += f"jobtype             sp\n"
        new_com_content += f"method              adc(2)\n"
        new_com_content += f"N_FROZEN_CORE       {N_FROZEN_CORE}\n"
        new_com_content += f"N_FROZEN_VIRTUAL    {N_FROZEN_VIRTUAL}\n"
        new_com_content += "basis               cc-pVDZ\n"
        new_com_content += "aux_basis           rimp2-cc-pVDZ\n"
        new_com_content += "mem_total           64000\n"
        new_com_content += "mem_static          1000\n"
        new_com_content += "maxscf              1000\n"
        new_com_content += "cc_symmetry         false\n"
        new_com_content += "ee_singlets         3\n"
        new_com_content += "ee_triplets         3\n"
        new_com_content += "sym_ignore          true\n"
        new_com_content += "ADC_DAVIDSON_MAXITER 300\n"
        new_com_content += "ADC_DAVIDSON_CONV 5\n"
        new_com_content += "$end\n"

        # Create the respective folder in the target directory
        target_subfolder = os.path.join(target_folder, folder_name)
        os.makedirs(target_subfolder, exist_ok=True)

        # Write the new all1.com file
        new_com_file_path = os.path.join(target_subfolder, "all1.com")
        with open(new_com_file_path, 'w') as f:
            f.write(new_com_content)

# Main function to process all folders in the source directory
def process_all_folders(source_base, target_base):
    for folder_name in os.listdir(source_base):
        folder_path = os.path.join(source_base, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing {folder_name}...")
            process_xyz_file(source_base, target_base, folder_name)

# Define source and target folder paths
source_folder = "/home/atreyee/BNPAH/top82_wB97XD3_opt_freq"  # Replace with the actual path
target_folder = "/home/atreyee/BNPAH/top82_wB97XD3_opt_freq/ADC2_FC_FV_VDZ"  # Replace with the actual path

# Process all folders
process_all_folders(source_folder, target_folder)

print("All folders processed successfully!")
```
#
```
import os
import shutil

# Define paths
thiol_smiles_folder = '/path/to/THIOL_smiles'  # Update this path
destination_folder = '/home/atreyee/THIOL/OPT_wB97XD3_def2SVP/molecules'

# Process each subfolder in the 'THIOL_smiles' folder
for subfolder in os.listdir(thiol_smiles_folder):
    subfolder_path = os.path.join(thiol_smiles_folder, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Find the 'foldername_conf.xyz' file inside the subfolder
        xyz_file = os.path.join(subfolder_path, f'{subfolder}_conf.xyz')
        
        if os.path.exists(xyz_file):
            print(f'Processing {subfolder}...')

            # Create corresponding folder in destination
            molecule_folder = os.path.join(destination_folder, subfolder)
            os.makedirs(molecule_folder, exist_ok=True)

            # Open the geometry file and start processing molecules
            with open(xyz_file, 'r') as geom_file:
                mol_counter = 1  # Initialize molecule counter

                line = geom_file.readline().strip()
                while line:
                    # Read number of atoms (Nat)
                    Nat = int(line)

                    # Read molecule title
                    title = geom_file.readline().strip()

                    # Create new molecule name (sequentially numbered)
                    mol_name = f"Mol_{mol_counter:05d}"
                    print(f'Processing {mol_name} for {subfolder}')

                    # Create folder for the molecule
                    mol_folder = os.path.join(molecule_folder, mol_name)
                    os.makedirs(mol_folder, exist_ok=True)

                    # Prepare new geom.xyz file
                    new_geomfile = os.path.join(mol_folder, 'geom.xyz')
                    with open(new_geomfile, 'w') as inputfile:
                        inputfile.write(f'{Nat}\n')
                        inputfile.write(f'{mol_name}\n')

                        # Read atoms' data for the molecule
                        for _ in range(Nat):
                            line = geom_file.readline().split()
                            sym = line[0]
                            R = [float(line[1]), float(line[2]), float(line[3])]
                            inputfile.write(f'{sym}   {R[0]:15.8f}   {R[1]:15.8f}   {R[2]:15.8f}\n')

                    # Copy necessary files into the new folder
                    shutil.copy('tddft.com', mol_folder)  # Adjust path if necessary

                    mol_counter += 1  # Increment molecule counter

                    # Read next molecule
                    line = geom_file.readline().strip()

        else:
            print(f'Warning: {xyz_file} not found in {subfolder}. Skipping...')
```
#
```
import csv

# Conversion factor from Hartree to kcal/mol
hartree_to_kcal = 627.509474

# Constants
H = -0.501445095782
SH = -398.62824952

# File paths
mol_file = "mol_energy.txt"
mol_C_file = "mol_C_energy.txt"
mol_S_file = "mol_S_energy.txt"
output_file = "results.csv"

# Function to read energy values from a file
def read_energies(file_path):
    energies = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                mol_name = parts[0]
                energy = float(parts[1])
                energies[mol_name] = energy
    return energies

# Read energy values from files
mol_energies = read_energies(mol_file)
mol_C_energies = read_energies(mol_C_file)
mol_S_energies = read_energies(mol_S_file)

# Perform calculations for each molecule and prepare results
results = []
for mol_name in mol_energies:
    mol = mol_energies[mol_name]
    mol_C = mol_C_energies[mol_name]
    mol_S = mol_S_energies[mol_name]
    
    # Calculate E1 and E2 in kcal/mol
    E1_hartree = mol - (mol_S + H)
    E2_hartree = mol - (mol_C + SH)
    E1_kcal = E1_hartree * hartree_to_kcal
    E2_kcal = E2_hartree * hartree_to_kcal
    
    # Append results as a tuple
    results.append((mol_name, E1_kcal, E2_kcal))

# Write results to a CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["molname", "E1", "E2"])
    # Write data
    writer.writerows(results)

print(f"Results saved to {output_file}")
```
#
```
import csv
import matplotlib.pyplot as plt

def plot_histogram_from_csv(file_path, output_file):
    labels = []
    data = []

    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(row[reader.fieldnames[0]])  # First column for labels
            data.append(float(row[reader.fieldnames[1]]))  # Second column for values

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Histogram of 2nd Column')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1)  # Add horizontal dotted line
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()

# Example usage
file_path = 'your_file.csv'  # Replace with your CSV file path
output_file = 'histogram.png'  # Replace with your desired output file path
plot_histogram_from_csv(file_path, output_file)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file = "data.csv"  # Replace with your CSV file name
data = pd.read_csv(csv_file)

# Extract the 1st, 4th, and 5th columns
labels = data.iloc[:, 0]  # 1st column (molecule labels)
x_values = data.iloc[:, 3]  # 4th column
y_values = data.iloc[:, 4]  # 5th column

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color="blue", alpha=0.7)

# Annotate points with labels
for label, x, y in zip(labels, x_values, y_values):
    plt.text(x, y, str(label), fontsize=9, ha='right', va='bottom')

# Add labels and title
plt.xlabel("4th Column Values", fontsize=12)
plt.ylabel("5th Column Values", fontsize=12)
plt.title("Scatter Plot of 4th Column vs 5th Column with Labels", fontsize=14)
plt.grid(True)

# Save the plot as a PNG file
output_file = "scatter_plot.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Scatter plot saved as {output_file}")
```
#
```
import os
import subprocess
import shutil

def process_folders():
    base_dir = os.getcwd()  # Get the current working directory
    all_csv_dir = os.path.join(base_dir, "all_csv_files")

    # Create 'all_csv_files' directory if it doesn't exist
    if not os.path.exists(all_csv_dir):
        os.makedirs(all_csv_dir)

    # Iterate over all folders in the base directory
    for folder in os.listdir(base_dir):
        if os.path.isdir(folder) and (folder.startswith('A') or folder.startswith('E') or folder.startswith('L')):
            # Find the .sh file starting with 'e' in the folder
            folder_path = os.path.join(base_dir, folder)
            e_sh_file = None
            for file in os.listdir(folder_path):
                if file.startswith('e') and file.endswith('.sh'):
                    e_sh_file = file
                    break
            
            if e_sh_file:
                # Run the script and redirect output to a CSV file
                result_file = os.path.join(folder_path, f"{folder}.csv")
                e_sh_path = os.path.join(folder_path, e_sh_file)
                with open(result_file, 'w') as outfile:
                    subprocess.run(['bash', e_sh_path], cwd=folder_path, stdout=outfile, stderr=outfile)

                # Copy the CSV file to all_csv_files folder
                shutil.copy(result_file, all_csv_dir)
                print(f"CSV file for {folder} copied to {all_csv_dir}")
            else:
                print(f"No .sh file starting with 'e' found in {folder}")

# Run the function
if __name__ == "__main__":
    process_folders()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file = "results.csv"  # Replace with your CSV file name
data = pd.read_csv(csv_file)

# Extract the required columns
labels = data.iloc[:, 0]  # 1st column (molecule labels)
x_values = data.iloc[:, 4]  # 4th column
y_values = data.iloc[:, 3]  # 5th column

# Coordinates for the lines
x_coords = [x_values.iloc[2], x_values.iloc[0], x_values.iloc[3], x_values.iloc[1]]  # 3rd to 1st, 4th to 2nd
y_coords = [y_values.iloc[2], y_values.iloc[0], y_values.iloc[3], y_values.iloc[1]]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color="blue", alpha=0.7)

# Annotate points with labels
for label, x, y in zip(labels, x_values, y_values):
    plt.text(x, y, str(label), fontsize=9, ha='right', va='bottom')

# Draw lines connecting points
plt.plot(x_coords[:2], y_coords[:2], color='red', linestyle='--', linewidth=1.5, label='Line: 3rd to 1st row')
plt.plot(x_coords[2:], y_coords[2:], color='green', linestyle='--', linewidth=1.5, label='Line: 4th to 2nd row')

# Add labels and title
plt.ylabel("Yield", fontsize=12)
plt.xlabel("Relative Stabilization Energy per System [kcal/mol]", fontsize=12)
plt.grid(True)
plt.legend()

# Save the figure as PNG
plt.tight_layout()
plt.savefig("scatter_plot_with_lines.png", dpi=300)  # Save as a high-quality PNG

# Show the plot
plt.show()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file = "results.csv"  # Replace with your CSV file name
data = pd.read_csv(csv_file)

# Extract the required columns
labels = data.iloc[:, 0]  # 1st column (molecule labels)
x_values = data.iloc[:, 4]  # 4th column
y_values = data.iloc[:, 3]  # 5th column

# Coordinates for the lines
x_coords = [x_values.iloc[2], x_values.iloc[0], x_values.iloc[3], x_values.iloc[1]]  # 3rd to 1st, 4th to 2nd
y_coords = [y_values.iloc[2], y_values.iloc[0], y_values.iloc[3], y_values.iloc[1]]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color="blue", alpha=0.7)

# Annotate points with labels
for label, x, y in zip(labels, x_values, y_values):
    plt.text(x, y, str(label), fontsize=9, ha='right', va='bottom')

# Draw lines connecting points
plt.plot(x_coords[:2], y_coords[:2], color='red', linestyle='--', linewidth=1.5, label='Line: 3rd to 1st row')
plt.plot(x_coords[2:], y_coords[2:], color='green', linestyle='--', linewidth=1.5, label='Line: 4th to 2nd row')

# Add labels and title
plt.ylabel("Yield", fontsize=12)
plt.xlabel("Relative Stabilization Energy per System [kcal/mol]", fontsize=12)
# plt.title("Scatter Plot with Lines Connecting Specific Points", fontsize=14)
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
```
#
```
import os
import shutil

def create_folders_and_copy_files(stable_file, source_folder, destination_folder):
    # Read the folder names from stable_41.txt
    with open(stable_file, 'r') as file:
        folders = [line.strip() for line in file.readlines()]
    
    # Create 41 folders like Mol_00001 to Mol_00041
    os.makedirs(destination_folder, exist_ok=True)
    for i in range(1, 42):
        folder_name = f"Mol_{i:05d}"
        folder_path = os.path.join(destination_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Copy the geom_DFT_S0.xyz from the source folders
        if i <= len(folders):
            source_geom_file = os.path.join(folders[i - 1], "geom_DFT_S0.xyz")
            
            # Copy geom_DFT_S0.xyz
            if os.path.isfile(source_geom_file):
                shutil.copy(source_geom_file, folder_path)
                print(f"Copied {source_geom_file} to {folder_path}")
            else:
                print(f"geom_DFT_S0.xyz not found in {folders[i - 1]}")
        
        # Copy the tddft.com file from the SOS-PBE-QIDH_AVDZ folder
        source_tddft_file = os.path.join(source_folder, "tddft.com")
        if os.path.isfile(source_tddft_file):
            shutil.copy(source_tddft_file, folder_path)
            print(f"Copied {source_tddft_file} to {folder_path}")
        else:
            print(f"tddft.com not found in {source_folder}")

# Usage
stable_file = 'stable_41.txt'  # Path to the stable_41.txt
source_folder = 'SOS-PBE-QIDH_AVDZ'  # Folder where tddft.com is located
destination_folder = 'SOS-PBE-QIDH_AVDZ'  # Destination folder

create_folders_and_copy_files(stable_file, source_folder, destination_folder)
```
#
```
import os

# Template for the inp.com file
inp_template = """memory,8,g
charge=0

gdirect
symmetry,nosym;orient,noorient

geometry={{
{coordinates}
}}

basis={{
default,avdz
set,mp2fit
default,avdz/mp2fit
set,jkfit
default,avdz/jkfit }}

hf

{{lt-df-lcc2                     !ground state CC2
eom,-3.1,triplet=1              !triplet states
eomprint,popul=-1,loceom=-1 }}   !minimize the output same thing for cc2 inp
"""

# Source and destination folders
source_base_path = "/home/atreyee/BNPAH/Pyrene_63_wB97XD3_def2TZVP_OPT_freq"
destination_base_path = "/home/atreyee/BNPAH/Pyrene_63_LCC2_AVDZ"

def format_coordinates(lines):
    """Formats the coordinates for alignment."""
    formatted_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) == 4:
            atom = parts[0]
            x, y, z = map(float, parts[1:])
            formatted_lines.append(f"{atom:<2} {x:>20.14f} {y:>20.14f} {z:>20.14f}")
    return "\n".join(formatted_lines)

def process_folders():
    # Walk through the source base path
    for root, dirs, _ in os.walk(source_base_path):
        for dir_name in dirs:
            source_folder = os.path.join(root, dir_name)
            destination_folder = os.path.join(destination_base_path, dir_name)

            # Ensure destination folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Check for the geom_DFT_S0.xyz file in the source folder
            geom_file = os.path.join(source_folder, "geom_DFT_S0.xyz")
            if os.path.exists(geom_file):
                # Read the geometry from the geom_DFT_S0.xyz file
                with open(geom_file, "r") as f:
                    lines = f.readlines()
                    # Skip the first two lines and format the remaining coordinates
                    coordinates = format_coordinates(lines[2:])

                # Create inp.com with the formatted coordinates
                inp_content = inp_template.replace("{coordinates}", coordinates)

                # Write inp.com to the destination folder
                inp_file_path = os.path.join(destination_folder, "inp.com")
                with open(inp_file_path, "w") as inp_file:
                    inp_file.write(inp_content)

                print(f"Created inp.com in {destination_folder}")
            else:
                print(f"geom_DFT_S0.xyz not found in {source_folder}")

# Run the function
process_folders()
```
#
```
import os
import subprocess
import shutil

def process_folders():
    base_dir = os.getcwd()  # Get the current working directory
    all_csv_dir = os.path.join(base_dir, "all_csv_files")

    # Create 'all_csv_files' directory if it doesn't exist
    if not os.path.exists(all_csv_dir):
        os.makedirs(all_csv_dir)

    # Iterate over all folders in the base directory
    for folder in os.listdir(base_dir):
        if os.path.isdir(folder) and (folder.startswith('A') or folder.startswith('E') or folder.startswith('L')):
            # Find the .sh file starting with 'e' in the folder
            folder_path = os.path.join(base_dir, folder)
            e_sh_file = None
            for file in os.listdir(folder_path):
                if file.startswith('e') and file.endswith('.sh'):
                    e_sh_file = file
                    break
            
            if e_sh_file:
                # Run the script and redirect output to a CSV file
                result_file = os.path.join(folder_path, f"{folder}.csv")
                e_sh_path = os.path.join(folder_path, e_sh_file)
                with open(result_file, 'w') as outfile:
                    subprocess.run(['bash', e_sh_path], cwd=folder_path, stdout=outfile, stderr=outfile)

                # Copy the CSV file to all_csv_files folder
                shutil.copy(result_file, all_csv_dir)
                print(f"CSV file for {folder} copied to {all_csv_dir}")
            else:
                print(f"No .sh file starting with 'e' found in {folder}")

# Run the function
if __name__ == "__main__":
    process_folders()
```
#
```
import numpy as np
import pandas as pd

def load_column(filename, col_index=2):
    """Loads the specified column (0-based index) from a CSV file without headers."""
    return pd.read_csv(filename, usecols=[col_index], header=None).values.flatten()

def compute_errors(reference, target):
    """Computes MSE, MAE, and SDE between reference and target arrays."""
    mse = np.mean(reference - target)
    mae = np.mean(np.abs(reference - target))
    sde = np.std(reference - target)
    return mse, mae, sde

# File names
reference_file = "Pyrene_63_LCC2_AVDZ.csv"
target_files = [
    "Pyrene_63_PBE-QIDH_AVDZ.csv",
    "Pyrene_63_PBE-QIDH_AVDZ_0.70_0.65.csv",
    "Pyrene_63_PBE-QIDH_AVDZ_0.75_0.45.csv"
]

# Load reference column
reference_data = load_column(reference_file)

# Compute and print errors for each target file
for target_file in target_files:
    target_data = load_column(target_file)
    mse, mae, sde = compute_errors(reference_data, target_data)
    print(f"Comparison with {target_file}:")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, SDE: {sde:.6f}\n")
```
#
```
import pandas as pd

# File names
files = [
    "LCC2_AVTZ.csv",
    "LADC2_AVDZ.csv",
    "Method_PBE-QIDH_AVDZ.csv",
    "Scaled_Method_PBE-QIDH_AVDZ.csv",
    "TBE.csv"
]

# Read CSV files and extract the 3rd column
columns = []
for file in files:
    data = pd.read_csv(file, header=None)
    columns.append(data.iloc[:12, 2].values)  # Extract first 12 values from the 3rd column

# Generate LaTeX table
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{c|ccccc}\n"
latex_table += "Serial & LCC2 & LADC2 & PBE-QIDH & Scaled PBE-QIDH & TBE \\\\ \hline\n"
for i in range(12):
    row = [str(i + 1)] + [f"{val:.3f}" for val in [col[i] for col in columns]]
    latex_table += " & ".join(row) + " \\\\ \n"
latex_table += "\\end{tabular}\n\\caption{Comparison of Methods}\n\\end{table}"

# Save to file
with open("table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved as table.tex")
```
#
```
import os
import shutil
import glob

def extract_gibbs_energy(folder):
    """Extract the final Gibbs free energy from opt_int.out"""
    file_path = os.path.join(folder, "opt_int.out")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        energies = [line for line in lines if "Final Gibbs free energy" in line]
        if energies:
            return float(energies[-1].split()[5])  # Extract energy value
    return None

def get_ps_c3_distance(xyz_file):
    """Extract the P-S and C(3)-S distances from geom_DFT_S0.xyz"""
    if not os.path.exists(xyz_file):
        return float('inf'), float('inf')
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    atoms = [line.split() for line in lines[2:]]  # Skip first two lines (header)
    p_atoms = [list(map(float, atom[1:])) for atom in atoms if atom[0] == "P"]
    s_atoms = [list(map(float, atom[1:])) for atom in atoms if atom[0] == "S"]
    
    # Ensure we are selecting the 4th C atom (index 3)
    c_atoms = [list(map(float, atom[1:])) for atom in atoms if atom[0] == "C"]
    
    if len(c_atoms) < 4:  # Ensure there are at least 4 carbon atoms
        return float('inf'), float('inf')
    
    c3_atom = c_atoms[3]  # Select the 4th C atom (C(3))
    
    if not p_atoms or not s_atoms or c3_atom is None:
        return float('inf'), float('inf')
    
    ps_distance = min(
        ((p[0] - s[0])**2 + (p[1] - s[1])**2 + (p[2] - s[2])**2)**0.5
        for p in p_atoms for s in s_atoms
    )
    c3s_distance = min(
        ((c3_atom[0] - s[0])**2 + (c3_atom[1] - s[1])**2 + (c3_atom[2] - s[2])**2)**0.5
        for s in s_atoms
    )
    
    return ps_distance, c3s_distance

def main():
    mol_folders = sorted(glob.glob("Mol_*"))
    energy_data = []

    for folder in mol_folders:
        energy = extract_gibbs_energy(folder)
        xyz_path = os.path.join(folder, "geom_DFT_S0.xyz")
        ps_distance, c3s_distance = get_ps_c3_distance(xyz_path)
        
        if energy is not None and ps_distance < 3.0 and c3s_distance < 3.0:
            energy_data.append((folder, energy))

    # Sort by energy (ascending order)
    energy_data.sort(key=lambda x: x[1])
    top_10 = energy_data[:10]

    # Create target folder
    target_folder = "OPT_10_wb97XD3_SVP"
    os.makedirs(target_folder, exist_ok=True)
    
    # Create and write to energy file
    energy_file_path = os.path.join(target_folder, "lowest_energies.txt")
    with open(energy_file_path, "w") as energy_file:
        for folder, energy in top_10:
            energy_file.write(f"{folder} {energy}\n")
    
    # Create combined XYZ file
    combined_xyz_path = os.path.join(target_folder, "top10.xyz")
    with open(combined_xyz_path, "w") as combined_xyz:
        for folder, _ in top_10:
            src_xyz = os.path.join(folder, "geom_DFT_S0.xyz")
            if os.path.exists(src_xyz):
                with open(src_xyz, "r") as xyz_file:
                    contents = xyz_file.readlines()
                combined_xyz.writelines(contents)

    for i, (folder, _) in enumerate(top_10, start=1):
        new_folder_name = f"Mol_{i:05d}"
        new_folder_path = os.path.join(target_folder, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Copy files
        src_xyz = os.path.join(folder, "geom_DFT_S0.xyz")
        src_com = os.path.join(target_folder, "opt.com")
        
        if os.path.exists(src_xyz):
            shutil.copy(src_xyz, new_folder_path)
        if os.path.exists(src_com):
            shutil.copy(src_com, new_folder_path)

    print("Process completed. 10 lowest energy molecules with P-S and C(3)-S distances < 3 Å copied, recorded in lowest_energies.txt, and combined in top10.xyz.")

if __name__ == "__main__":
    main()
```
#
```
import os
import glob
import csv

def extract_gibbs_energy(folder):
    """Extract the final Gibbs free energy from opt_int.out"""
    file_path = os.path.join(folder, "opt_int.out")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        energies = [line for line in lines if "Final Gibbs free energy" in line]
        if energies:
            return float(energies[-1].split()[5])  # Extract energy value in Hartree
    return None

def hartree_to_kcalmol(energy_hartree):
    """Convert energy from Hartree to kcal/mol"""
    return energy_hartree * 627.509

def main():
    mol_folders = sorted(glob.glob("Mol_*"))
    energy_data = []

    for folder in mol_folders:
        energy_hartree = extract_gibbs_energy(folder)
        if energy_hartree is not None:
            energy_kcalmol = hartree_to_kcalmol(energy_hartree)
            energy_data.append((folder, energy_kcalmol))

    # Save results to CSV
    csv_filename = "gibbs_energies.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Folder Name", "Energy (kcal/mol)"])
        writer.writerows(energy_data)

    print(f"Saved Gibbs free energies in {csv_filename}")

if __name__ == "__main__":
    main()
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the first column (labels) and second column (values)
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    
    # Convert reference energy from Hartree to kcal/mol
    reference_energy_hartree = -720.79682962
    reference_energy_kcal = reference_energy_hartree * 627.509
    
    # Subtract reference energy
    adjusted_values = values - reference_energy_kcal
    
    # Adjust scale for better visualization
    plt.figure(figsize=(10, 6))
    plt.bar(labels, adjusted_values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.ylim(min(adjusted_values) - 1, max(adjusted_values) + 1)  # Zoom in further
    plt.xlabel('Conformer Names')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    csv_file = "Cys_lowest_10_energies.csv"  # Replace with your actual CSV file name
    plot_histogram(csv_file)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the first column (labels) and second column (values)
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    
    # Conversion factor from Hartree to kcal/mol
    hartree_to_kcal = 627.509
    
    # Convert reference energies from Hartree to kcal/mol
    reference_energy_1 = -720.79682962 * hartree_to_kcal
    reference_energy_2 = -1024.51562513 * hartree_to_kcal
    
    # Subtract both reference energies sequentially
    adjusted_values = values - reference_energy_1 - reference_energy_2
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, adjusted_values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.ylim(min(adjusted_values) - 1, max(adjusted_values) + 1)  # Zoom in further
    plt.xlabel('Conformer Names')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    csv_file = "Cys_lowest_10_energies.csv"  # Replace with your actual CSV file name
    plot_histogram(csv_file)
```
#
```
systems=$( echo Ac_Cys Ac_pen Cys Pen )

for sys in $systems; do
  cd $sys
  rm -f gibbs.txt

  # Initialize variables to track the folder with the lowest energy
  min_energy=9999999  # Set a high initial value for comparison
  min_folder=""

  # Determine the file pattern based on the folder name
  if [[ $sys == "Ac_Cys" || $sys == "Ac_pen" ]]; then
    file_pattern="Mol*/opt1.out"
  else
    file_pattern="Mol*/opt.out"
  fi

  for mol in $file_pattern; do
    energy=$(grep 'Final Gibbs free energy' $mol | tail -1 | awk '{print $6}')
    folder_name=$(basename $(dirname $mol))  # Extract the folder name (Mol?)

    # Update if the current energy is lower than the previous minimum
    if (( $(echo "$energy < $min_energy" | bc -l) )); then
      min_energy=$energy
      min_folder=$folder_name
    fi
  done

  # Print the system, minimum energy, and folder with the lowest energy
  echo "$sys: Lowest energy = $min_energy in folder $min_folder"
  cd ..
done
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the first column (labels) and second column (values)
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    
    # Adjust scale for better visualization
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.ylim(min(values) - 1, max(values) + 1)  # Zoom in further
    plt.xlabel('Folder Name')
    plt.ylabel('Energy (kcal/mol)')
    plt.title(f'Histogram of {df.columns[1]} by {df.columns[0]}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    csv_file = "your_file.csv"  # Replace with your actual CSV file name
    plot_histogram(csv_file)
```
#
```
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the first column (labels) and second column (values)
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    
    # Conversion factor from Hartree to kcal/mol
    hartree_to_kcal = 627.509
    
    # Convert reference energies from Hartree to kcal/mol
    reference_energy_1 = -720.79682962 * hartree_to_kcal
    reference_energy_2 = -1024.51562513 * hartree_to_kcal
    
    # Subtract both reference energies sequentially
    adjusted_values = values - reference_energy_1 - reference_energy_2
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, adjusted_values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.ylim(min(adjusted_values) - 1, max(adjusted_values) + 1)  # Zoom in further
    plt.xlabel('Conformer Names')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    csv_file = "Cys_lowest_10_energies.csv"  # Replace with your actual CSV file name
    plot_histogram(csv_file)
```
#
```
import csv

def correct_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            corrected_value = round(float(row[2]) * 0.9546 - 0.0065, 3)
            writer.writerow(row + [corrected_value])

# Example usage
correct_csv('input.csv', 'corr.csv')
```
#
```
lowest_energy=9999999  # Initialize with a large number
lowest_folder=""

for sys in */; do
  # Determine the file pattern based on the folder name
  if [[ $sys == "Ac_Cys" || $sys == "Ac_pen" ]]; then
    file_pattern="Mol*/opt1.out"
  else
    file_pattern="Mol*/opt.out"
  fi

  gibbs_file="gibbs.txt"
  > $gibbs_file  # Clear gibbs.txt before appending new data

  for mol in $file_pattern; do
    grep 'Final Gibbs free energy' $mol | tail -1 >> $gibbs_file
  done

  energy=$( sort -k6 $gibbs_file | tail -1 | awk '{print $6}' )
  echo "$sys $energy"

  # Check if the current folder has the lowest energy
  if (( $(echo "$energy < $lowest_energy" | bc -l) )); then
    lowest_energy=$energy
    lowest_folder=$sys
  fi

  cd ..
done

# Print the folder with the lowest energy
echo "Folder with the lowest energy: $lowest_folder with energy $lowest_energy"
```
#
```
import csv
import matplotlib.pyplot as plt

def plot_histogram_from_csv(file_path, output_file):
    labels = []
    data = []

    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(row[reader.fieldnames[0]])  # First column for labels
            data.append(float(row[reader.fieldnames[1]]))  # Second column for values

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Histogram of 2nd Column')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1)  # Add horizontal dotted line
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()

# Example usage
file_path = 'your_file.csv'  # Replace with your CSV file path
output_file = 'histogram.png'  # Replace with your desired output file path
plot_histogram_from_csv(file_path, output_file)
```
#
```
import numpy as np
import pandas as pd

def compute_errors(file1, file2):
    # Load CSV files
    data1 = np.loadtxt(file1, delimiter=',', usecols=(0, 1, 2))  # Method data
    data2 = np.loadtxt(file2, delimiter=',', usecols=(0, 1, 2))  # TBE data
    
    # Compute error metrics
    error = data1 - data2
    minE = np.min(error, axis=0)
    maxE = np.max(error, axis=0)
    mse = np.mean(error, axis=0)
    mae = np.mean(np.abs(error), axis=0)
    sde = np.std(error, axis=0)
    
    # Method name
    method = file1.replace('.csv', '')
    
    # Print results in Overleaf table format
    energies = ["S$_1$", "T$_1$", "STG"]
    print(f"{method}$^a$              &{energies[0]}&  {mse[0]:8.3f} &  {mae[0]:8.3f} &  {sde[0]:8.3f} &  {minE[0]:8.3f} &  {maxE[0]:8.3f}\\")
    print(f"                     &{energies[1]}&  {mse[1]:8.3f} &  {mae[1]:8.3f} &  {sde[1]:8.3f} &  {minE[1]:8.3f} &  {maxE[1]:8.3f}\\")
    print(f"                     &{energies[2]}&  {mse[2]:8.3f} &  {mae[2]:8.3f} &  {sde[2]:8.3f} &  {minE[2]:8.3f} &  {maxE[2]:8.3f}\\")

# Example usage
compute_errors("LCC2_AVDZ.csv", "TBE.csv")
```
#
```
import numpy as np
import matplotlib.pyplot as plt

# Load data (assuming no headers)
def load_column(filename, col_index=2):
    return np.loadtxt(filename, delimiter=',', usecols=[col_index])

# Load 3rd column from each file (index 2 in zero-based indexing)
tbe_x = load_column("TBE.csv")
adc2_y = load_column("LCC2_AVTZ.csv")
adc2_tbe_y = load_column("CC2_AVTZ.csv")

# Uncomment this line if the file exists
# cc2_y = load_column("LADC2_AVTZ.csv")

# Ensure the arrays have the same length
if len(tbe_x) != len(adc2_y) or len(tbe_x) != len(adc2_tbe_y):
    raise ValueError("Data columns have different lengths!")

# Plot scatter plots
plt.figure(figsize=(8, 8))  # Square plot
plt.scatter(tbe_x, adc2_y, label="L-CC2/aug-cc-pVTZ (This work)", color='r', marker='x', alpha=0.7)

# Only plot CC2 if the file exists
# plt.scatter(tbe_x, cc2_y, label="CC2/aug-cc-pVTZ (Ref 1)", color='b', marker='s', alpha=0.7)

plt.scatter(tbe_x, adc2_tbe_y, label="ADC(2)/aug-cc-pVTZ (Ref 1)", color='g', marker='o', alpha=0.7)

# Plot y = x line
plt.plot([-0.45, 0.0], [-0.45, 0.0], linestyle='--', color='black')

# Set limits
plt.xlim(-0.45, 0.0)
plt.ylim(-0.45, 0.0)

# Labels and legend
plt.xlabel("S$_1$-T$_1$ gap, TBE (eV)")
plt.ylabel("S$_1$-T$_1$ gap, CC2 (eV)")
plt.legend()
plt.grid(True)

# Set font
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'sans-serif'  # Use default sans-serif instead of Arial

# Save as PDF
plt.savefig("scatter_plot_s1_2.pdf", dpi=300, bbox_inches='tight')

plt.show()
```
