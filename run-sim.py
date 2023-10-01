import batsim_py
from schedulers.FCFSScheduler import FCFSScheduler
from schedulers.EASYScheduler import EASYScheduler

def run_simulation(scheduler, platform_path: str, workload_path: str):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    jobs_mon = batsim_py.monitors.JobMonitor(simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform=platform_path, workload=workload_path, verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        # Check reward
        simulator.proceed_time() # proceed directly to the next event.

    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon

jobs_f, sim_f = run_simulation(FCFSScheduler,
                               "/data/platforms/FatTree/generated.xml",
                               "/data/workloads/example_workload_hpc_seed4_jobs250.json")
                               #"/data/workloads/test_batsim_paper_workload_seed1.json")

jobs_f.to_csv(f"/data/expe-out/jobs-FCFS.out")
sim_f.to_csv("/data/expe-out/sim-FCFS.out")
