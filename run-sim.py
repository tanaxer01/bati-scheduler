import batsim_py
from model.metrics import MonitorsInterface
from envs.shutdown_policies import TimeoutPolicy
from schedulers.FCFSScheduler import FCFSScheduler
from schedulers.EASYScheduler import EASYScheduler

def run_simulation(scheduler, platform_path: str, workload_path: str):
    simulator = batsim_py.SimulatorHandler()
    policy = TimeoutPolicy(1, simulator)
    scheduler = scheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    monitors = MonitorsInterface(
            name = str(scheduler),
            save_dir ="/data/expe-out",
            monitors_fns = [
                batsim_py.monitors.JobMonitor,
                batsim_py.monitors.SimulationMonitor,
                batsim_py.monitors.SchedulerMonitor,
                batsim_py.monitors.ConsumedEnergyMonitor,
                batsim_py.monitors.HostStateSwitchMonitor
            ])
    monitors.init_episode(simulator, True)

    # 2) Start simulation
    simulator.start(platform=platform_path, workload=workload_path, verbosity="quiet")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        # Check reward
        simulator.proceed_time() # proceed directly to the next event.

    simulator.close()

    # 4) Return/Dump statistics
    monitors.record()


run_simulation(FCFSScheduler,
               "/data/platforms/FatTree/generated.xml",
               "/data/workloads/test/w.json")

run_simulation(EASYScheduler,
               "/data/platforms/FatTree/generated.xml",
               "/data/workloads/test/w.json")

