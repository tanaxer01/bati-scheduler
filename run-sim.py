import batsim_py
from schedulers.FCFSScheduler import FCFSScheduler

def get_reward() -> float:
    qos = 0.

    return 0.

def run_simulation(scheduler, platform_path: str, workload_path: str):
    simulator = batsim_py.SimulatorHandler()
    scheduler = scheduler(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    sched_mon = batsim_py.monitors.SchedulerMonitor(simulator)
    jobs_mon = batsim_py.monitors.JobMonitor(simulator)
    sim_mon  = batsim_py.monitors.SimulationMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform=platform_path, workload=workload_path, verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        scheduler.schedule()
        # Check reward
        print(get_reward)
        simulator.proceed_time() # proceed directly to the next event.

    simulator.close()

    # 4) Return/Dump statistics
    return sched_mon, jobs_mon, sim_mon

sched_f, jobs_f, sim_f = run_simulation(FCFSScheduler,
                               "/data/platforms/FatTree/generated.xml",
                               "/data/workloads/test_batsim_paper_workload_seed1.json")

sched_f.to_csv("/data/expe-out/sched.out")
jobs_f.to_csv("/data/expe-out/jobs.out")
sim_f.to_csv("/data/expe-out/sim.out")

