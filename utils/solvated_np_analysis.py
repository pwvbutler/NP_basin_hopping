import random
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

import matplotlib.pyplot as plt


# def extract_np_atoms_from_simulation_cell(atoms: Atoms, np_atom_symbols: list):
#     extracted_np_atoms = Atoms()

#     for atom in atoms:
#         if atom.symbol in np_atom_symbols:
#             extracted_np_atoms += atom
            
#     centre_point = np.mean(extracted_np_atoms.positions, axis=0)
#     extracted_np_atoms.positions -= centre_point
#     extracted_np_atoms.set_pbc(False)

#     return extracted_np_atoms

def extract_np_atoms_from_simulation_cell(atoms: Atoms, np_atom_symbols: list):
    np_atom_indices = []

    for i, atom in enumerate(atoms):
        if atom.symbol in np_atom_symbols:
            np_atom_indices.append(i)

    extracted_np_atoms = atoms[np_atom_indices]
    # centre_point = np.mean(extracted_np_atoms.positions, axis=0)
    # extracted_np_atoms.positions -= centre_point
    # extracted_np_atoms.set_pbc(False)

    return extracted_np_atoms

def plot_elemental_radial_distribution(np_atoms):
    centre_point = np.mean(np_atoms.positions, axis=0)
    radial_distances = { symb: [] for symb in set(np_atoms.get_chemical_symbols()) }
    for atom in np_atoms:
        symbol = atom.symbol
        radial_dist = np.linalg.norm(atom.position - centre_point)
        radial_distances[symbol].append(radial_dist)

    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,7))
    for symbol, data in radial_distances.items():
        counts, bin_edges = np.histogram(data, bins=30)
        cumulative = np.cumsum(counts )#* np.diff(bin_edges))  # Cumulative sum
        bin_edges = np.insert(bin_edges, 0, bin_edges[0])  # Extend the bin edges
        cumulative = np.insert(cumulative, 0, 0)
        ax.plot(bin_edges[1:], cumulative, label=symbol)

    ax.set_xlabel("radial distance, Ang")
    ax.set_ylabel("cumulative num atoms")
    fig.legend(loc=(0.2, 0.8))
    fig.tight_layout()


def calculate_composition(np_atoms):
    counts = { symb: 0 for symb in set(np_atoms.get_chemical_symbols()) }
    total = 0
    for atom in np_atoms:
        counts[atom.symbol] += 1
        total += 1
    
    composition =  {}
    for symb, count in counts.items():
        composition[symb] = {
            "count": count, 
            "percent": (count / total)*100,
        }
    
    return composition

def extract_surface_atoms(np_atoms, distance_ratio_threshold=0.65):
    # Create a NeighborList for the structure
    cutoffs = natural_cutoffs(np_atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(np_atoms)

    # Calculate the geometric center of the structure
    geometric_center = np.mean(np_atoms.get_positions(), axis=0)

    # Calculate the distances of all atoms from the geometric center
    distances = [np.linalg.norm(atom.position - geometric_center) for atom in np_atoms]

    # Identify the center atoms as the 5 atoms closest to the geometric center
    center_atoms = np.argsort(distances)[:5]

    # Calculate the mean number of neighbors for the center atoms
    mean_neighbors_center_atoms = np.mean([len(nl.get_neighbors(index)[0]) for index in center_atoms])

    # Identify the surface atoms as those that have fewer neighbors than the center atoms
    surface_atoms = [atom.index for atom in np_atoms if len(nl.get_neighbors(atom.index)[0]) < mean_neighbors_center_atoms-0]

    # Calculate distances from each atom to the center of mass
    atom_positions = [np_atoms[index].position for index in surface_atoms]
    atom_distances_from_center = np.linalg.norm(atom_positions - geometric_center, axis=1)

    # Set a threshold for being considered 'far from center', e.g., beyond a certain ratio of the max distance
    distance_threshold = distance_ratio_threshold * np.max(atom_distances_from_center)

    # Filter surface atoms based on the distance constraint
    surface_atoms = [atom for atom, distance in zip(surface_atoms, atom_distances_from_center) if distance > distance_threshold]

    return np_atoms[surface_atoms]


def get_contact_atoms(atoms, np_atom_symbols):
    np_atom_idxs = [ atom.index for atom in atoms if atom.symbol in np_atom_symbols ]

    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    contact_idxs = []  
    for idx in np_atom_idxs:
        indices, _ = nl.get_neighbors(idx)
        contacts = [ i for i in indices if atoms[i].symbol not in np_atom_symbols ]
        contact_idxs.extend(contacts)

    contact_atoms = atoms[[atom.index for atom in atoms if atom.index in contact_idxs]]

    counts = { symb: 0 for symb in set(contact_atoms.get_chemical_symbols()) }

    for atom in contact_atoms:
        counts[atom.symbol] += 1

    return contact_atoms, counts

def get_contact_type_counts(atoms, np_atom_symbols):
    np_atom_idxs = [ atom.index for atom in atoms if atom.symbol in np_atom_symbols ]

    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    contacts = []  
    for idx in np_atom_idxs:
        np_symb = atoms[idx].symbol
        indices, _ = nl.get_neighbors(idx)
        contact_idxs = [ i for i in indices if atoms[i].symbol not in np_atom_symbols ]
        
        contacts.extend([f"{np_symb}-{atoms[i].symbol}" for i in contact_idxs])

    return {x: contacts.count(x) for x in set(contacts)}

def plot_surface_composition_convergence(traj, np_atom_symbols, title=None):
    """
    plot convergence of nanoparticle surface composition

    Params:
        traj - [ase.Atoms]
            the lowest energy structures found
        np_atom_symbols - [str]
            the chemical symbols of the atoms in the nanoparticle
    """
    surface_comp = { symb: [] for symb in np_atom_symbols }
    for atoms in traj:
        np_atoms = extract_np_atoms_from_simulation_cell(atoms, np_atom_symbols)
        comp = calculate_composition(extract_surface_atoms(np_atoms))
        for symb in surface_comp:
            if symb in comp:
                surface_comp[symb].append(comp[symb]["percent"])
            else:
                surface_comp[symb].append(0.0)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,7))

    for symb, data in surface_comp.items():
        ax.step(data, label=symb)

    if title is not None:
        ax.set_title(title)
        
    ax.set_xlabel("lowest energy structure")
    ax.set_ylabel("surface composition, %")
    fig.legend(loc=(0.2, 0.8))
    fig.tight_layout()

    return fig, ax

def plot_surface_composition_convergence_against_step(traj, steps, np_atom_symbols, final_step=2500, title=None):
    """
    plot convergence of nanoparticle surface composition

    Params:
        traj - [ase.Atoms]
            the lowest energy structures found
        steps - [int]
            the step number corresponding to each structure in the trajectory
        np_atom_symbols - [str]
            the chemical symbols of the atoms in the nanoparticle
    """

    assert len(traj) == len(traj), "steps and trajectory not same length"

    surface_comp = { symb: [] for symb in np_atom_symbols }
    for atoms in traj:
        np_atoms = extract_np_atoms_from_simulation_cell(atoms, np_atom_symbols)
        comp = calculate_composition(extract_surface_atoms(np_atoms))
        for symb in surface_comp:
            if symb in comp:
                surface_comp[symb].append(comp[symb]["percent"])
            else:
                surface_comp[symb].append(0.0)

    if final_step not in steps:
        steps.append(final_step)
        for symb in surface_comp:
            surface_comp[symb].append(surface_comp[symb][-1])
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,7))

    for symb, data in surface_comp.items():
        ax.step(steps, data, label=symb)

    if title is not None:
        ax.set_title(title)
        
    ax.set_xlabel("step")
    ax.set_ylabel("surface composition, %")
    fig.legend(loc=(0.2, 0.8))
    fig.tight_layout()

    return fig, ax