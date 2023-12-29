from rdkit import Chem

def count_molecules_in_sdf(sdf_file):
    """
    Counts the number of molecules in an SDF file.

    Args:
    - sdf_file: Path to the input SDF file.

    Returns:
    - count: Number of molecules in the SDF file. Returns -1 if unable to read the file.
    """
    count = 0
    try:
        suppl = Chem.SDMolSupplier(sdf_file)
        for mol in suppl:
            if mol is not None:
                count += 1
    except Exception as e:
        print(f"An error occurred: {e}")
        count = -1

    return count

def write_specific_molecule_to_sdf(input_sdf, output_sdf, mol_index=0):
    """
    Reads an SDF file, extracts a specific molecule by index, and writes it to an output SDF file.

    Args:
    - input_sdf: Path to the input SDF file.
    - output_sdf: Path to the output SDF file to write the specific molecule.
    - mol_index (optional): Index of the molecule to extract. Default is 0.
    
    Returns:
    - True if the molecule is successfully extracted and written, False otherwise.
    """
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
    mol = None
    for idx, m in enumerate(suppl):
        if idx == mol_index:
            mol = m
            break

    if mol is None:
        return False

    writer = Chem.SDWriter(output_sdf)
    writer.write(mol)
    writer.close()
    return True

def extract_atom_info_from_conformer(input_sdf, mol_index=0, conf_id=0):
    """
    Extracts atomic coordinates and names of atoms for a specified conformer from an SDF file.

    Args:
    - input_sdf: Path to the input SDF file.
    - mol_index (optional): Index of the molecule in the SDF file. Default is 0.
    - conf_id (optional): ID of the conformer to extract coordinates from. Default is 0.

    Returns:
    - atom_info: List of tuples containing atomic names and coordinates for the specified conformer.
                 Returns None if the molecule or conformer ID is invalid or no conformers exist.
    """
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
    mol = None
    for idx, m in enumerate(suppl):
        if idx == mol_index:
            mol = m
            break

    if mol is None or conf_id < 0 or conf_id >= mol.GetNumConformers():
        return None

    conformer = mol.GetConformer(conf_id)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    atom_info = []
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_name = atom.GetSymbol()
        pos = conformer.GetAtomPosition(atom_idx)
        coords = [pos.x, pos.y, pos.z]
        atom_info.append(['UNK', atom_name]+coords) # unkown residue name, we dont care

    return atom_info