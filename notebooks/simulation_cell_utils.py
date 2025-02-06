import random
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import molecule
from ase.geometry import get_distances
from ase.neighborlist import NeighborList, natural_cutoffs


def identify_water_molecules(atoms):
    waters = []
    for i, atom in enumerate(atoms):
        if atom.symbol == 'O':
            # Find the two nearest H atoms
            distances = atoms.get_distances(i, range(len(atoms)), mic=True)
            h_indices = [j for j in range(len(atoms)) if atoms[j].symbol == 'H' and distances[j] < 1.2]
            if len(h_indices) == 2:
                waters.append([i] + h_indices)
    return waters

def check_clashes(atoms, new_molecule, threshold=2.0):
    combined = atoms + new_molecule
    cutoffs = natural_cutoffs(combined)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(combined)
    for i, atom in enumerate(new_molecule):
        indices, offsets = nl.get_neighbors(len(atoms) + i)
        for idx in indices:
            if idx < len(atoms) and (combined.get_distance(idx, len(atoms) + i) < threshold):
                return True
    return False

def get_clashes(atoms, new_molecule):
    combined = atoms + new_molecule
    cutoffs = natural_cutoffs(combined)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(combined)
    clashes = []
    for i, atom in enumerate(new_molecule):
        indices, offsets = nl.get_neighbors(len(atoms) + i)
        for idx in indices:
            if idx < len(atoms): #and (combined.get_distance(idx, len(atoms) + i) < threshold):
                clashes.append(idx)
    return clashes

def randomly_rotate_solvent_shell_molecules(shell_structure):
    waters = identify_water_molecules(shell_structure)
    new_waters = Atoms()
    for water in waters:
        water_atoms = shell_structure[water].copy()
        com = water_atoms.get_center_of_mass()
        water_atoms.translate(-com)
        water_atoms.rotate(random.uniform(0, 180), 'x')
        water_atoms.rotate(random.uniform(0, 180), 'y')
        water_atoms.rotate(random.uniform(0, 180), 'z')
        water_atoms.translate(+com)
        new_waters += water_atoms

    new_shell_structure = shell_structure.copy()
    new_shell_structure = new_shell_structure[[atom.index for atom in new_shell_structure if atom.index not in [x for water in waters for x in water]]]
    new_shell_structure += new_waters

    return new_shell_structure

def substitute_n_water_molecules(atoms, solute, n):
    waters = identify_water_molecules(atoms)
    added_positions = []
    n_added = 0
    new_atoms = atoms.copy()
    attempts = 0

    while n_added < n:
        attempts += 1
        solute_copy = solute.copy()
        water = random.choice(waters)
        water_pos = new_atoms[water[0]].position
        print("placing mol {} at {}".format(n_added+1, water_pos))
        solute_copy.translate(water_pos - solute_copy.get_center_of_mass())

        # Randomly rotate the ethanol molecule
        solute_copy.rotate(random.uniform(0, 360), 'x')
        solute_copy.rotate(random.uniform(0, 360), 'y')
        solute_copy.rotate(random.uniform(0, 360), 'z')

        if attempts > 200:
            print("unable to add any more solutes")
            break


        # Combine all positions for clash checking
        if len(added_positions) > 0:
            # Check for clashes
            dm = get_distances(solute_copy.positions,  np.vstack(added_positions), cell=new_atoms.get_cell(), pbc=True)[1]
            if np.any(dm < 2.4):  # Adjust threshold if necessary
                print("clash with added atoms (smallest dist {}), trying again".format(np.min(dm)))
                continue  # Skip this placement if there's a clash

        clashes = get_clashes(new_atoms, solute_copy)
        
        try:
            to_remove = []
            for idx in clashes:
                to_remove.extend(*[x for x in waters if idx in x])
        except:
            print("clash with non water atoms")
            continue

        new_atoms = new_atoms[[atom.index for atom in new_atoms if atom.index not in to_remove]]
        new_atoms += solute_copy
        added_positions.extend(solute_copy.positions)
        waters = identify_water_molecules(new_atoms)
        n_added += 1
        attempts = 0
    
    return new_atoms


def add_solute_to_solvent_box(solvent_box, solute_atoms, position, solvent_n_atoms):
    """assumes solvent atoms are grouped by molecule"""
    new_solute = solute_atoms.copy()
    new_solute.positions += position
    clashes = get_clashes(solvent_box, new_solute)

    to_remove = []
    for idx in clashes:
        solvent_mol_idx = idx // solvent_n_atoms
        start_idx = solvent_mol_idx * solvent_n_atoms
        end_idx = start_idx + solvent_n_atoms
        to_remove += [ x for x in range(start_idx, end_idx)]

    simulation_cell = solvent_box.copy()
    simulation_cell = simulation_cell[[atom.index for atom in simulation_cell if atom.index not in to_remove]]
    simulation_cell += new_solute
    

    return simulation_cell
    


def create_solvent_shell_structure(solvent_box, ref_np_atoms, np_position, solvent_n_atoms, radial_cutoff=16.5, ):
    """assumes solvent atoms are grouped by molecule"""
    np_atoms = ref_np_atoms.copy()
    np_atoms.positions += np_position
    clashes = get_clashes(solvent_box, np_atoms)

    to_remove = []
    for idx in clashes:
        solvent_mol_idx = idx // solvent_n_atoms
        start_idx = solvent_mol_idx * solvent_n_atoms
        end_idx = start_idx + solvent_n_atoms
        to_remove += [ x for x in range(start_idx, end_idx)]

    
    for solvent_start_idx in range(0, len(solvent_box), solvent_n_atoms):
        solvent_idxs = [ x for x in range(solvent_start_idx, solvent_start_idx + solvent_n_atoms)]
        com = solvent_box[solvent_idxs].get_center_of_mass()
        if np.linalg.norm(com - np_position) > radial_cutoff:
            to_remove += solvent_idxs
    
    

    simulation_cell = solvent_box.copy()
    simulation_cell = simulation_cell[[atom.index for atom in simulation_cell if atom.index not in to_remove]]
    simulation_cell += np_atoms
    simulation_cell.set_pbc(False)
    simulation_cell.set_cell(None)

    simulation_cell.positions -= np_position
    

    return simulation_cell

def calculate_density(atoms):
    mass_unit_cell_grams = sum(atoms.get_masses()) * 1.66054e-24
    volume_unit_cell_cm3 = atoms.get_volume() * 1e-24

    return mass_unit_cell_grams / volume_unit_cell_cm3
    
