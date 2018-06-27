# WHAM: Weighted histogram analysis method
Code to do a binless WHAM (aka multistate Bennett acceptance ratio / MBAR). The code is flexible enough that, given a series of umbrella sampling windows along one reaction coordinate, can project to another reaction coordinate (assuming the reaction coordinates overlap sufficiently). It can also combine different umbrella sampling simulations performed with differen reaction coordnates (perhaps you were trying a  few different reaction coordinates) to make a "super" PMF.

The input requires a `metadata` file, which essentially is the same as used in Alan Grossfield's WHAM code, except there is an additional argument indicating which column of data corresponds to the reaction coordinate the restraining force was applied to. 

In other words, each row of the `metadata` file goes:

```
# (path to file)     (float)              (float)          (integer)
trajectory_file_1 restraint_location_1 force_constant_1 reaction_coordinate_index_1
trajectory_file_2 restraint_location_2 force_constant_2 reaction_coordinate_index_2
...
trajectory_file_n restraint_location_n force_constant_n reaction_coordinate_index_n
```

and each `trajectory_file` has the following columns:

```
Time trajectory_along_coordinate_1 trajectory_along_coordinate_2 ...trajectory_along_coordinate_n
```

It should go without saying that the trajectory file columns should be consistent among each other.

## Dependencies
`numpy` and `matplotlib` (if you want to plot)

## Usage
Given a `metadata` and trajectory files available like above, here's a sample script:

```
from wham import WHAM

wham = WHAM(T=300.0,metadata='metadata')
wham.compute_free()
wham.compute_pmf(hmin=-2.0,hmax=2.0,num_bins=40,pmf_crd=2)
```

which would compute the free energies from the umbrella sampling simulation defined along the reaction coordinate in your `metadata` file, then once those are computed it would build up the histogram from `-2.0` to `2.0` with 40 bins along the reaction coordinate in the second column of your trajectory data files. 

You can also save and load free energies, to either avoid re-computing the unbiasing, or to compare with free energies obtained from another WHAM program.

```
wham = WHAM(T=300.0,metadata='metadata')
wham.compute_free(save_free_energies='./myFreeEnergies.txt')
wham.compute_pmf(hmin=-2.0,hmax=2.0,num_bins=40,pmf_crd=2,load_free_energies='./myFreeEnergies.txt')
```

would save the free energies computed in `compute_free()` to `./myFreeEnergies.txt` and then load them for use in plotting the PMF in `compute_pmf()`. Default save and load location is in the current folder: `./free-energies.txt`. 
