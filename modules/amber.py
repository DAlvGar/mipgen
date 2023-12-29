# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="dalvarez"
__date__ ="$18-feb-2011 0:14:26$"

import re
import os
import sqlite3
import subprocess as sub
from os.path import split, splitext, exists
import numpy as npy
from modules.RDKit import extract_atom_info_from_conformer, write_specific_molecule_to_sdf

def parse_mol2_file(file_path):
    atomic_info = []
    reading_atoms = False
    mol_name = 'MOL'
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('@<TRIPOS>ATOM'):
                reading_atoms = True
                continue
            elif line.startswith('@<TRIPOS>'):
                reading_atoms = False

            if reading_atoms and line:
                parts = line.split()
                atom_data = [
                    parts[1], #'atom_name'
                    parts[5], # 'atom_type'
                    float(parts[8]), #'charge'
                    float(parts[2]), #'x': 
                    float(parts[3]), #'y': 
                    float(parts[4]), #'z': 
                    parts[7] #'residue': 
                ]
                
                atomic_info.append(atom_data)

    return {mol_name:atomic_info}

def readAmberPrep(prepfile):
    """
    Read an Amber prepin file and return the residues, atom names, atom types,\
    and charges for each atom in a dictionary {RESNAME:[[atname1,attype1,charge1], [..],..], ...}
    """
    res = re.compile(r'\w+\s+INT\s+')       # Begin of a residue in the prepin file
    molec = {}

    f = open(prepfile, 'r')
    l = f.readline()
    while(l):
        l = f.readline()
        if res.search(l):
            resname = l.split()[0].strip()
            molec[resname] = []
            #skip 2 lines
            f.readline()
            f.readline()

            # begins atom section
            s = f.readline().strip()
            while(bool(s)):
                atline = s.split()
                atname, attype, charge = atline[1], atline[2], atline[-1]
                if atname != 'DUMM': molec[resname].append((atname, attype, charge))
                s = f.readline().strip()

    f.close()
    return molec

class Leaper:
    """Handles tLeap I/O"""
    def __init__(self,  cwd=None):
        """Initiates a suprocess connection with tLeap program from Ambertools package"""
        self.tleap = sub.Popen("tleap -f - ",  shell=True,   stdin = sub.PIPE,  stdout=sub.PIPE,  stderr=sub.PIPE, cwd=cwd)
        self._in = self.tleap.stdin
        self._out = self.tleap.stdout
        self._err = self.tleap.stderr

    def command(self,  command):
        """
        Command must be a string corresponding to a correct Leap command.
        Returns the output of leap for the given command.
        tLeap is a bit tricky to be controlled by stdin because it terminates when it reaches EOF when reading STDOUT.
        We use a fake command to stop reading STDOUT right before the EOF is reached.
        """
        self._in.write(command+'\n')
        self._in.write('lastCommandOut\n')
        out = []
        while 1:
            line = self._out.readline().strip()
            if 'ERROR: syntax error' in line: break
            if line: out.append(line)
        return out

    def close(self):
        self.tleap.terminate()
        del self.tleap

class AmberMolecule:
    """
    This class generates molecule instance with all the atoms parametrized using Amber ForceField.
    If the molecule is a small organic compound, antechamber is run to calculate partial charges,
    then GAFF ForceField atomtypes are assigned.
    """
    def __init__(self, inFile, mode, charge=0, mol_ix=0, charge_method='bcc'):
        """Build an instance for the molecule containing
        for each atom, the coordinates and non-bonded parameters
        (mass, charge, vdw radii and epsilon)

        Input:  inFile - PDB file or SDF file
                mode - 'gaff' or 'parm'
                charge - net charge
                mol_ix - from the sdf, which molecule to read
        """
        
        self.atoms = self.__getParamsXYZ(inFile, mode, charge, mol_ix, charge_method)
        self.natoms = len(self.atoms)
        self.xyz = npy.array([at[3:] for at in self.atoms])


    def __getParamsXYZ(self, inputFile, mode, charge=0, mol_ix=0, charge_method='bcc'):
        """
        Obtain for each atom the correspondent parameters for
        Vander Waals and Coulomb calculations along with their coordinates.
        This is done using an external software: AmberTools v.1.1.

        For organic molecules, GAFF force field should be used, and PARM99
        is used for proteins atom type identification.

        Thus two modes can be chosen in the mode parameter of the function:

        Mode = 'gaff'   --> Organic molecule. Treat with antechamber.
        Mode = 'parm'   --> Protein. Treat with tLeap and then assign Amber FF params.

        charge = 0      --> Molecule net charge, default is zero.

        The function returns a list with this format:
        [[charge, vdw, eps, x, y, z],[..],..]
        """
        
        # If its a protein, 'parm' mode, first give a try to normalize a bit
        # the file using tLeap from AmberTools. It happens that the PDB can
        # have different occupancies and/or extrange names. This will filter a bit
        # this abnormalities
        if mode == 'parm':
            pdb = self.__fixPDBtLeap(inputFile)
            if pdb: inputFile = pdb
            else: pass # Will continue without pre-process and good luck!

        # GAFF is used for organic molecules. Antechamber is used to prepare them.
        # PARM99 is used for proteins. Parameters are obtained from amber.db
        if mode == 'gaff':
            # First obtain atom types and charges
            types_charges_xyz = self.__gaffParams(inputFile, charge, mol_ix, charge_method)
            if types_charges_xyz is None:
                raise RuntimeError("Antechamber calcualtion failed")
            # res_at_xyz = self.__getAtsXYZ(inputFile, mol_ix)
            xyz = [[at[3],at[4],at[5]] for at in types_charges_xyz]

            # Assign corresponding VdW parameters using gaff.db
            params = self.__addVdWtoGaff(types_charges_xyz)
            return [params[i]+xyz[i] for i in range(len(params))]

        elif mode == 'parm':
            # First extract from the PDB the coordinates, atom names and residue
            # name for each atom
            res_at_xyz = self.__getAtsXYZ(inputFile)
            res_at = [[at[0],at[1]] for at in res_at_xyz]
            params, miss = self.__amberParams(res_at)
            xyz = [at[2:] for at in res_at_xyz if at[0:2] not in miss]
            self.miss = miss
            return [params[i]+xyz[i] for i in range(len(params))]

    def __queryDB(self, sqliteConnection, DB, attype=None, atname=None, resname=None):
        "Query database amber.db or gaff.db by atom name or by atom type and return result.\
        DB should be either 'amber' or 'gaff'."

        if DB == 'gaff':
            table = 'gafftypes'
        elif DB == 'amber':
            table = 'ambertypes'
        else:
            return False

        # Create a pointer
        c = sqliteConnection.cursor()

        # Prepare where clause:
        if atname and resname:
            where = "resname='%s' and atname='%s'"%(resname, atname)
        elif attype:
            where = "attype='%s'"%attype

        # Query it!
        c.execute("SELECT * FROM %s WHERE %s"%(table,where))
        row=c.fetchall()
        if len(row) != 1:
            # No results
            print('ATTENTION! Missing atom:')
            print("At:",atname,resname,attype)
            return 'miss',resname,atname
        else:
            # Correct fetching
            return row[0]

    def __addVdWtoGaff(self,types_charges):
        "Add the corresponding VdW parameters from gaff.db to the given atomtypes"

        # Open sqlite connection and query the db for each atom type
        conn = sqlite3.connect('parm/gaff.db')
        # TODO if some atom is missing, catch the error!
        vdw=[self.__queryDB(conn, 'gaff', attype=at[1])[2:] for at in types_charges]
        conn.close()

        # Return a list Charge,VdWRadii,Eps
        return [[float(types_charges[i][2])]+list(vdw[i]) for i in range(len(types_charges))]


    def __fixPDBtLeap(self, inputFile):
        "Pre-process PDB using AmberTools software. Just load it and save it again.\
        It will change some atom names and some residues hopefully fixing some strange things.\
        Will also add missing Hydrognes."
        outname = inputFile.replace('.','_tleap.')
        tleap = Leaper()
        tleap.command('p = loadPdb %s'%inputFile)
        tleap.command('savePdb p %s'%outname)
        tleap.close()

        if exists(outname):
            return outname
        else:
            return False

    def __getAtsXYZ(self,inputFile,mol_ix=0):
        "Given a pdb file, return the list of residue names and atoms names for each atom."
        if '.sd' in splitext(inputFile)[1]:
            # Not PDB, treat as SDF file
            molec = extract_atom_info_from_conformer(inputFile, mol_ix, 0) # first molecule, first conformer
        else:
            from modules.PDBParser import readResAtCoordFromPDB
            molec = readResAtCoordFromPDB(inputFile)
        return molec

    def __amberParams(self, res_at):
        """Returns for each pair residue_atomname a tuple (charge, radii, epsilon)
        Using Amber 99 FF"""
        # Open sqlite connection and query the db for each atom type
        conn = sqlite3.connect('parm/amber.db')
        # TODO if some atom is missing, catch the error!
        params = []
        miss = []
        for at in res_at:
            if at[1] == 'OXT': at[0] = 'C'+at[0]
            if at[0] == 'HIS': at[0] = 'HIE'
            p = self.__queryDB(conn, 'amber', resname=at[0], atname=at[1])
            if p[0] != 'miss': params.append(list(p)[4:])
            else: miss.append(p[1:])
        conn.close()
#        print(params
        # Return a list Charge,VdWRadii,Eps
        return params, miss

    def __gaffParams(self, molFile, charge=0, mol_ix=0, charge_method='bcc'):
        """Returns for each atom in pdbInstance a tuple (charge, radii, epsilon, x, y, z)
        Using Amber GAFF FF."""
        basename, ext = splitext(molFile)
        outfile = None
        if 'sd' in ext: 
            ext = 'mdl' # sdf is mdl format in antechamber
            outfile = basename +'_'+ str(mol_ix) +'.sd'
            write_specific_molecule_to_sdf(molFile, outfile, mol_ix)
            molFile = outfile
            mol2 = basename +'_'+ str(mol_ix) +'.mol2'
        elif 'pdb' in ext or 'ent' in ext: 
            ext = 'pdb'
            mol2 = basename+'.prep'
        else: raise TypeError
        ante_cmd = "antechamber -i %s -fi %s -o %s -fo mol2 -c %s -pf y -nc %i"%(molFile, ext, mol2, charge_method, charge)

        print(f"Running antechamber to calculate charges and assign atomtypes {mol_ix}...")
        if not exists(mol2):
            antechamber = sub.Popen(ante_cmd, shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
            antechamber.wait()
#            antechamber.terminate()

        if exists(mol2):
            #molec = readAmberPrep(mol2)
            molec = parse_mol2_file(mol2)
            # Will return only first residue as it should only contain 1 :)
            if outfile: os.remove(outfile) # no need to keep it
            return molec[list(molec.keys())[0]]
        else:
            print(antechamber.stdout.read())
            print(antechamber.stderr.read())
            return None


    def __getitem__(self, idx):
        return self.atoms[idx]


if __name__ == "__main__":
    import sys
    molec = readAmberPrep(sys.argv[1])
    for res in molec.keys():
        print(res, molec[res])
