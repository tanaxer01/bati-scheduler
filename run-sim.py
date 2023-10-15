import batsim_py
from schedulers.FCFSScheduler import FCFSScheduler
from schedulers.EASYScheduler import EASYScheduler

def run_simulation(scheduler, platform_path: str, workload_path: str):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    jobs_mon = batsim_py.monitors.JobMonitor(simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(simulator)
    schedule_mon  = batsim_py.monitors.SchedulerMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform=platform_path, workload=workload_path, verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        # Check reward
        simulator.proceed_time() # proceed directly to the next event.

    simulator.close()

    # 4) Return/Dump statistics
    return jobs_mon, sim_mon, schedule_mon

jobs_f, sim_f, schedule_f = run_simulation(FCFSScheduler,
                               "/data/platforms/FatTree/generated.xml",
                               "/data/workloads/test/w.json")

jobs_f.to_csv(f"/data/expe-out/jobs-FCFS.out")
sim_f.to_csv("/data/expe-out/sim-FCFS.out")
schedule_f.to_csv("/data/expe-out/schedule-FCFS.out")


jobs_e, sim_e, schedule_e = run_simulation(EASYScheduler,
                               "/data/platforms/FatTree/generated.xml",
                               "/data/workloads/test/w.json")

jobs_e.to_csv(f"/data/expe-out/jobs-EASY.out")
sim_e.to_csv("/data/expe-out/sim-EASY.out")
schedule_e.to_csv("/data/expe-out/schedule-EASY.out")

