from typing import IO, Type, Union, List, Optional

import numpy as np

from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.optimize.fire import FIRE
from ase.optimize.optimize import Dynamics, Optimizer
from ase.parallel import world
from ase.neighborlist import NeighborList, natural_cutoffs


class ElementBasinHopping(Dynamics):
    """Basin hopping algorithm with MC moves that swap neighboring atom types

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(
        self,
        atoms: Atoms,
        swap_indices: Optional[list] = None,
        swap_elements: Optional[list] = None,
        swap_different: bool = False,
        temperature: float = 100 * units.kB,
        optimizer: Type[Optimizer] = FIRE,
        fmax: float = 0.1,
        logfile: Union[IO, str] = '-',
        trajectory: str = 'lowest.traj',
        optimizer_logfile: str = '-',
        local_minima_trajectory: str = 'local_minima.traj',
        adjust_cm: bool = True,
    ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: string
            Trajectory file used to store optimisation path.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        """
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.swap_different = swap_different
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile
        self.lm_trajectory = local_minima_trajectory
        if isinstance(local_minima_trajectory, str):
            self.lm_trajectory = self.closelater(
                Trajectory(local_minima_trajectory, 'w', atoms))

        # If the user did not supply a list, allow all atoms to be swapped.
        if swap_indices is None:
            swap_indices = [ i for i, _ in enumerate(atoms) ]
        self.swap_indices = swap_indices

        # If the user did not supply a list, allow all elements to be swapped.
        if swap_elements is None:
            swap_elements = list(set(atoms.get_chemical_symbols()))
        self.swap_elements = swap_elements

        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.initialize()

    def todict(self):
        d = {'type': 'optimization',
             'optimizer': self.__class__.__name__,
             'local-minima-optimizer': self.optimizer.__name__,
             'temperature': self.kT,
             'max-force': self.fmax,
             'swap-elements': self.swap_elements,
             'swap-different': self.swap_different,}
        return d

    def initialize(self):
        positions = self.optimizable.get_positions()
        atomic_numbers = self.optimizable.get_atomic_numbers()
        self.atomic_numbers = np.zeros_like(atomic_numbers)

        #self.Emin = self.get_energy(positions) or 1.e32
        self.Emin = self.get_energy(atomic_numbers) or 1.e32

        self.rmin = self.optimizable.get_positions()
        self.positions = self.optimizable.get_positions()
        self.call_observers()
        self.log(-1, self.Emin, self.Emin)

    def run(self, steps):
        """Hop the basins for defined number of steps."""

        configuration = self.atomic_numbers
        Eo = self.get_energy(configuration)

        for step in range(steps):
            En = None
            while En is None:
                new_configuration = self.move(configuration)
                if new_configuration is not None:
                    En = self.get_energy(new_configuration)

            if En < self.Emin:
                # new minimum found
                self.Emin = En
                self.best_configuration = self.optimizable.get_atomic_numbers()
                self.call_observers()
            self.log(step, En, self.Emin)

            accept = np.exp((Eo - En) / self.kT) > np.random.uniform()
            if accept:
                configuration = new_configuration
                Eo = En

    def log(self, step, En, Emin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f\n'
                           % (name, step, En, Emin))
        self.logfile.flush()

    def _atoms(self):
        from ase.optimize.optimize import OptimizableAtoms
        assert isinstance(self.optimizable, OptimizableAtoms)
        # Some parts of the basin code cannot work on Filter objects.
        # They evidently need an actual Atoms object - at least until
        # someone changes the code so it doesn't need that.
        return self.optimizable.atoms

    def move(self, configuration):
        """
        Attempt a Monte Carlo move that swaps atom types of
        one randomly selected swappable atom with one of its neighbors,
        provided that neighbor's atomic number is in the allowed list.
        """
        atoms = self._atoms()

        # Randomly pick an atom from the list of swappable indices
        i = np.random.choice(self.swap_indices)

        # Get neighbors of atom i from the neighbor list
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        indices, _ = nl.get_neighbors(i)

        if len(indices) == 0:
            return None # no neighbors, try new move

        # Pick a random neighbor
        j = np.random.choice(indices)

        # Check if neighbor's element is in swap_elements
        if atoms[j].symbol not in self.swap_elements:
            return None  # skip if that neighbor can't be swapped


        if self.swap_different and atoms[i].symbol == atoms[j].symbol:
            return None

        new_configuration = configuration[:]
        new_configuration[i], new_configuration[j] = new_configuration[j], new_configuration[i]

        atoms.set_atomic_numbers(new_configuration)
        new_configuration = atoms.get_atomic_numbers()
        world.broadcast(new_configuration, 0)
        atoms.set_atomic_numbers(new_configuration)

        return atoms.get_atomic_numbers()

    def get_minimum(self):
        """Return minimal energy and configuration."""
        atoms = self._atoms().copy()
        atoms.set_positions(self.rmin)
        return self.Emin, atoms

    def get_energy(self, atomic_numbers):
        """Return the energy of the nearest local minimum."""
        if np.any(self.atomic_numbers != atomic_numbers):
            self.atomic_numbers = atomic_numbers
            self.optimizable.set_atomic_numbers(atomic_numbers)

            with self.optimizer(self.optimizable,
                                logfile=self.optimizer_logfile) as opt:
                opt.run(fmax=self.fmax)
            if self.lm_trajectory is not None:
                self.lm_trajectory.write(self.optimizable)

            self.energy = self.optimizable.get_potential_energy()

        return self.energy
