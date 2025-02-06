import numpy as np

import ase.units as units
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase import io
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE
from ase.optimize.precon import PreconLBFGS
import sys, os

from mace.calculators import mace_mp, MACECalculator

fpath = sys.argv[1]

name = os.path.basename(fpath)[:-4]
atoms = io.read(fpath)

tag = name + '_opt'

#macemp = mace_mp(enable_cueq=True)
#macemp = mace_mp(dispersion=True, enable_cueq=True)
#macemp = MACECalculator(model_paths="/home/energy/patbu/basin_hopping/ft_mpa_0_stagetwo.model", dispersion=True, enable_cueq=True)
macemp = MACECalculator(model_paths="/home/energy/patbu/basin_hopping/dft_data/multihead_finetune/mpa/ft_solvated_mpa_0_medium_stagetwo.model", default_dtype="float64", dispersion=True, enable_cueq=True, device="cuda")
#macemp = mace_mp(model="small", dispersion=True)
#macemp = mace_mp()
atoms.calc = macemp

traj_writer = Trajectory(tag+'.traj', 'w', atoms)

#opt = PreconLBFGS(atoms)
#opt = BFGS(atoms)
opt = FIRE(atoms, trajectory=traj_writer)
opt.run(fmax=0.01)
#opt.run(fmax=0.005)


atoms.write(tag + ".xyz", format="extxyz")

