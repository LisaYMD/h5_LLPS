# readdy_LLPS
Small python packages for reading LLPS synapse simulation of readdy h5 file

## content
-- readdy_LLPS/
 ├── h5Cluster/
 |    ├── h5readCluster
 |    ├── h5readCluster2D
 |    ├── h5readCluster3D
 |    └── h5readCOM3D
 ├── h5Density/
 |    ├── h5readDensity
 |    ├── h5readFRAP
 |    └── h5readMSD
 └── h5Traj/
      ├── h5readTraj
      └── h5readCheckpoint

## how to install/uninstall
```
python3 setup.py develop # install
python3 setup.py develop -u # uninstall
```

## how to use
Import libraries by writing an example below.
```
from h5Traj import h5readCheckpoint
```
