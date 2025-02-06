import numpy as np

import ase.units as units
from ase import Atoms
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase import io
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE
from ase.optimize.precon import PreconLBFGS

from ase.optimize.element_basin import ElementBasinHopping

from mace.calculators import mace_mp
from mace.calculators import MACECalculator
import sys, os


fpath = sys.argv[1]
name = os.path.basename(fpath)[:-4]
atoms = io.read(fpath)
tag = name+'_bh'

macemp = mace_mp(model="medium", dispersion=True, enable_cueq=True, device="cuda", default_dtype="float64")
atoms.calc = macemp

traj_writer = Trajectory(tag+'.traj', 'w', atoms)

np_atom_symbols = ["Au", "Cu"]
np_indices = []
for i, atom in enumerate(atoms):
    if atom.symbol in np_atom_symbols:
        np_indices.append(i)

bh = ElementBasinHopping(
    atoms=atoms,         
    temperature=298 * units.kB, 
    swap_indices=np_indices,
    swap_elements=np_atom_symbols,
    swap_different=True,
    fmax=0.05,          
)

bh.run(steps=2500)
