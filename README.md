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
 - to run the program give the command
   ```
   python3 obminimize.py nmol geom.xyz geom_UFF.xyz
   ```
   where nmol is number of molecules
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
# To read the csv filenames from a file and to find the errors
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
# calculation of gap in partition coeff
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
