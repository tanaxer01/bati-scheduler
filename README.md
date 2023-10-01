# Testing scheduling schemes in Batsim [WIP]
The idea of this proyect is to explore diferent scheduling algorithms
and compare their performance in diferent environments.

## Build with

## Getting started
### Retrieving simulation inputs
We can retrieve the platform and workload files provided by Batsim to generate basic tests.

```bash
# Dowload a tarball of Batsim's latest release.
batversion='4.2.0'

curl --output "batsim-v${batversion}.tar.gz" \
    "https://framagit.org/batsim/batsim/-/archive/v${batversion}/batsim-v${batversion}.tar.gz"

# Extract tarball and copy important folders
(tar -xf "batsim-v${batversion}.tar.gz" && \
cp -r "batsim-v${batversion}/workloads" workloads && \
rm -r "batsim-v${batversion}" "batsim-v${batversion}.tar.gz"
cp -r "batsim-v${batversion}/platforms" platforms && \
```

### Preparing the simulation output directory
```bash
# Create the directory if needed
mkdir -p expe-out

# Clean the directory's content if any
rm -rf expe-out/*
```
### Build simulation environment
```bash
git clone https://github.com/tanaxer01/bati-scheduler
cd bati-scheduler
docker image build -t batsim-py .
```

### Executing the simulation
```bash
docker container run --rm -v .:/data batsim-py python /data/run-sim.py
```
