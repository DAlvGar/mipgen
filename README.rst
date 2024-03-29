======
MIPGEN
======

Author: Daniel Alvarez (algarcia.daniel@gmail.com)
Last revision date: 28-12-2023


1. INTRODUCTION
2. DEPENDENCIES AND INSTALLATION
    2.1 DEPENDENCIES
    2.2 INSTALLATION
3. USAGE
    3.1 INPUT FILES
    3.2 OPTIONS
    3.3 OUTPUT FILES
    3.4 PROBES
    3.5 FORCEFIELD DATABASE
    3.6 EXAMPLES
4. GRIDS VISULATIZATION WITH PYMOL
5. FUTURE WORK



1. INTRODUCTION
--------------------------------------------------------------------------------

MIPGen is a program used to calculate Molecular Interaction Potential
(MIP) grids. These grid files represent the favorable interacting regions
of a probe moiety over the space around some molecule.

The molecule could be either a protein/peptide chain, or some small organic 
compound (a ligand). The output of the program is a serie of DX format text files
which contain the energy in kcal/mol for each spatial point in the lattice.

These files can be visualized using any molecular visulaization program like
VMD, PyMol or Chimera i.e. See section 3.- GRIDS VISUALIZATION WITH PYMOL

This software is released under the GNU General Public License (GPL).





2. DEPENDENCIES AND INSTALLATION
--------------------------------------------------------------------------------

2.1  DEPENDENCIES
^^^^^^^^^^^^^^^^
The program was ported to Python 3 in DEC 2023. Newer versions should be 
also stable although no test was performed. Older python versions might
present some errors.

It uses standard built-in modules except for the ones listed below that
should be installed independently by the user:

* Numpy (http://numpy.scipy.org/)
* AmberTools 1.4 (http://ambermd.org/#AmberTools).
This software is REQUIRED for the parametrization of the small ligands (partial
charge calculation and atomtype identification using antechamber) and for the
cleaning of the PDB files (protonation of missing H, coorection of multiple
positions for same atom, etc.. using tLeap). Make sure that both programs: 'tLeap' and 'antechamber' are executable from any
folder in the system (add the folder containing the executables to the PATH
environmental variable). AmberTools is also released under GPL license.


2.2 INSTALLATION
^^^^^^^^^^^^^^^^^^
There is no real setup developed at the moment. Untar the file and reference to it.
After ensuring that all the dependencies are installed, simply unpack the tarball
to the folder of your choice and the software should be ready to run.

    `tar xzf mipgen_XXX.tar.gz`

It is recomendable to add an environmental variable pointing to the software
to facilitate execution in any folder of the system (in bash):

    `export MIPGEN=/installation/path/mipgen_XXX`

then executing the program in any folder like:

    `python $MIPGEN/MIPgen.py [options]`

or to PATH env:

    `export PATH=$PATH:/installation/path/mipgen_XXX`

and executing then:

    `MIPgen.py [options]`


3. USAGE
--------------------------------------------------------------------------------
The program is quite simple to run. You need an input file containing the molecule
over which the MIP grids will be calculated for different probes chosen. The
program will automatically identify the atom types and charges on the structure
and it will calculate the interaction energy for every grid point. The output will
be saved in ascii files with the .dx extension.

3.1. INPUT FILES
^^^^^^^^^^^^^^^^
The expected input files are of ascii PDB format (http://www.wwpdb.org/docs.html).
The program is prepared for two cases:

A) Calculate MIP grids over a protein

    In this case, the PDB file does not need previous modifications or
    preparation. If it does not have hydrogens, the program will try to add
    them using tLeap. In the same manner, if there are atoms with
    multiple occupancy, the program will choose the more occupied position.
    Histidines will be treated as HIE (proton in epsilon position) unless the
    user specifically change the name of the HIS to HID (proton in delta) or
    HIP (double protonated, charged histidines).

    In the same way, other specifications can be made over the PDB file 
    (like specifying protonated ASP or neutral LYS) following Amber
    Sotware manual.

    The correct parametrization of all the atoms depende entirely on the 
    user. If something is left unparametrized, the program will show a
    warning indicating what are the missing atoms.

    IMPORTANT: Do not let the program to guess everything
    if there are special considerations to take care of.

B) Calculate MIP grids over small organic compounds

    In this case, the molecule should contain all the hydrogens and
    all the atoms valences should be correctly set before running the
    program. Apart from that, antechamber should take good care of
    identifying all the atom types and setting the partial charges
    using AM1-BCC method.

3.2. OPTIONS
^^^^^^^^^^^^^^
To get a list of all the possible options for the program, type:

    `MIPgen.py -h or --help`

To get a list of probes available, type:

    `MIPgen.py -l`

When asking for help, you should get a list like this:

    Options:
      -h, --help            show this help message and exit
      -p PROT, --prot=PROT  Protein file (PDB format)
      -m MOLEC, --molec=MOLEC
                            Molecule file (PDB format)
      -r PROBES, --probe=PROBES
                            Append probe names to calculate the MIPs. This flag
                            can be used             more than once.
      -o OUT, --out=OUT     Output name prefix. Use some name descriptive of your
                            job. Default: MIP
      -L LIB, --lib=LIB     File containing the probes and its parameters.
                            Default: parm/probes.lib
      -E EPS, --eps=EPS     Relative permitivity for the electrostatic
                            calculations (float). If             0. is given, a
                            Distance Dependent Relative Permitivity is used
                            (default)
      -v VDW, --vdw=VDW     VdW calculations Cutoff (float). Default: 10A
      -e ELEC, --elec=ELEC  Electrostatic calculations cutoff (float). Default:
                            20A
      -l, --list            List available probes

All the flags are accompained with self explanatory information.
There are TWO mandatory flags:

    -m OR -p    --> Indicate the program if the input file is a protein or a
                    small molecule (as described in section 3.1)
    -r          --> Probe to use for calculating the MIP. This flag can be repeated
                    for multiple probes to be calculated in the same program call.

Optional RECOMMENDED flags:

    -o          --> Prefix for all the output files

Optional flags:

    -L          --> If given, this file should contain other probes defined
                    by the user (see section 3.4)
    -E          --> Relative permitivity of the medium. If zero, distance dependent
                    electrostatics will be applied (this is the default). If some
                    value is given here, permitivity is constant and independent on
                    the charges distance.
    -e          --> Cutoff distance in angstroms for the electrostatics calculation.
                    By default: 20 angstroms.
    -v          --> Cutoff distance in angstroms for van der Waals calculation.
                    By default: 10 angstroms.

3.3. OUTPUT FILES
^^^^^^^^^^^^^^^^^
The program will generate one file per probe chosen. The name of the file
will be:

    `prefix_PROBE.dx`     if `-o prefix` is given

or a default name:

    `MIP_PROBE.dx`        if `-o` not given as argument

where PROBE is the name of the probe.

These files are formatted in a way that almost all molecular visualization
programs will understand them, usually as a volume map or electron density
map files. An example on how to visualize this files with PyMol is given in
section 4.

3.4. PROBES
^^^^^^^^^^^^
The probes list with a short description can be obtained with:

`MIPgen.py -l`

The initial set contains a not very well tested set of parameters. This set
can be easily modified to add, remove or edit any probe. The file
containing the parameters can be found in parm/probes.lib. Add here any probe
or modify the parameters as you wish. It is also possible to generate any other
file with this same format and give it as argument (EXTRAPROBESFILE.txt).

`MIPgen.py -p XXX -o XXX -r EXTRAPROBE -L EXTRAPROBESFILE.txt`

The format should be as follows:

`PROBENAME   CHARGE  VDW_radii   VDW_EPSILON     DESCRIPTION`

Lines starting with # are ignored.

3.5. FORCEFIELD DATABASE
^^^^^^^^^^^^^^^^^^^^^^^^
Instead of using Amber Topology files, the program tries to identify the atom
types in the protein using a sqlite3 database (amber.db).
This database was generated and is stored in in parm/ folder.
The forcefield used was parm99 with amino03 modifications for proteins,
and GAFF forcefield for the small orgainc compounds.

For more details on how to generate the database with other amber forcefield
, please take a look at the generate*.py scripts in parm/ folder or contact
Daniel Alvarez.

3.6. EXAMPLES
^^^^^^^^^^^^^
In the test/ folder you will find two files: peptide.pdb and ligand.pdb
The former is a 3 peptide long structure representing a protein system.
The latter is a small organic compound (a commercialized drug: sustiva).

Here i provide some examples on how to run the program using those files:

I) Getting a hydrophobic MIP over the ligand, with distance dependent
    electrostatics and a long vanderwaals cutoff (15A)

    `MIPgen.py -m test/ligand.pdb -r HYD -o test_ligand -v 15`

II) Getting a hydrophobic, h-bond donor and h-bond acceptor MIP grid
    over the peptide, with a constant dielectric parameter of 8

    `MIPgen.py -p test/peptide.pdb -r HYD -r HDON -r HACC -o test_peptide -E 8`

III) Same as before with distance dependent electrostatic and long cutoff (25A)

    `MIPgen.py -p test/peptide.pdb -r HYD -r HDON -o test_peptide2 -e 25`

IV) Calculate the electrostatics for the ligand with default parameters (distance
    dependent electrostatics, cutoff 20A, vdW cutoff 10A):

    `MIPgen.py -m test/ligand.pdb -r POS -r NEG -o test_ligand_electr`

To visualize the resulting grids, jump to the next section ;)

4. GRIDS VISULATIZATION WITH PYMOL
--------------------------------------------------------------------------------
To follow this section, you will need to have installed PyMOL (a free software
copy is still available here http://sourceforge.net/projects/pymol/).

To demonstrate the usage of pymol for the grid visualization, run the last example
proposed (IV) and type this command in the shell:

    `pymol test/ligand.pdb test_ligand_electr_POS.dx test_ligand_electr_NEG.dx`

This should load the ligand.pdb and the 2 grid files. Now, on the right menu,
three objects should appear:

- ligand
- test_ligand_electr_POS
- test_ligand_electr_NEG

But only the molecule is visible on the main window. If you click on the second
object (test_ligand_electr_POS), the boundaries of the grid should appear on the image
with white lines. Same for any grid.

To diplay the content of the grid, choose an isovalue (i.e. -1 kcal/mol) and type
in the program shell:

    `isomesh positive, test_ligand_electr_POS, -1`

This will display a mesh for the isovalue -1. If we want to change the isovalue (-3)
,repeat the command above:

    `isomesh positive, test_ligand_electr_POS, -3`

Multiple representations can be produced changing the name of the mesh:

    `isomesh positive_2, test_ligand_electr_POS, -0.5`

To color them, click on the C button on the right of the object generated.
Color this grid in red.

Now diplay the negative potential:

    `isomesh negative, test_ligand_electr_NEG, -1`

and color in blue (click on the C besides test_ligand_electr_NEG and choose blue).
A nice image should appear on the main window displaying both potentials.


5. FUTURE WORK
--------------------------------------------------------------------------------

- Test thoroughly the probe parameters to better reproduce the expected behaviour.
- Introduce new probes.
- Allow the user to provide parameters for the 'missing atoms'.
- Include more precise calculations apart from electrostatics and vanderwaals.

6. CHANGELOG
-------------
- 12/2023: Use of KDTrees to significantly speed up calculations.
