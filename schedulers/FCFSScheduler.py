from batsim_py.events import JobEvent
from batsim_py.simulator import SimulatorHandler

class FCFSScheduler:
    def __init__(self, simulator: SimulatorHandler) -> None:
        self.simulator = simulator

        self.completed_jobs = set()
        self.simulator.subscribe(JobEvent.COMPLETED, self.on_job_completed)

    def __str__(self) -> str:
        return "FCFS"

    @property
    def valid_jobs(self):
        dependencies_met = lambda x: "real_subtime" in x.metadata or "dependencies" not in x.metadata

        # Jobs real subtime needs to take into account when they became available
        valid_jobs  = filter(lambda x: dependencies_met(x), self.simulator.queue)
        sorted_jobs = sorted(valid_jobs, key=lambda x: x.metadata["real_subtime"] if "dependencies" in x.metadata else x.subtime)
        return sorted_jobs

    def on_job_completed(self, job):
        self.completed_jobs.add( int(job.name) )

        child_nodes = filter(lambda j: "dependencies" in j.metadata and int(job.name) in j.metadata["dependencies"], self.simulator.jobs)
        for j in child_nodes:
            j.metadata["real_subtime"] = self.simulator.current_time

    def schedule(self) -> None:
        """  First Come First Served policy """
        assert self.simulator.is_running

        for job in self.valid_jobs:
            available = self.simulator.platform.get_not_allocated_hosts()

            if job.res <= len(available):
                # Schedule if the job can start now.
                allocation = [h.id for h in available[:job.res]]
                self.simulator.allocate(job.id, allocation)
            else:
                # Otherwise, wait for resources.
                break
