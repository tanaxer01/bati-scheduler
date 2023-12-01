
from typing import Dict
from batsim_py import HostEvent, SimulatorEvent, SimulatorHandler
from batsim_py.resources import Host

class TimeoutPolicy:
    def __init__(self, timeout: int, simulator: SimulatorHandler):
        self.timeout   = timeout
        self.simulator = simulator
        self.idle_servers : Dict[int, float] = {}

        super().__init__()

        self.simulator.subscribe(HostEvent.STATE_CHANGED, self._on_host_state_changed)
        self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self._on_sim_begins)

    def shutdown_idle_hosts(self, *args, **kwargs):
        hosts_to_turnoff = []
        for h_id, start_t in list(self.idle_servers.items()):
            if self.simulator.current_time - start_t >= self.timeout:
                hosts_to_turnoff.append(h_id)
                del self.idle_servers[h_id]

        if hosts_to_turnoff:
            print([ i.state for i in self.simulator.platform.hosts ])
            print([ i.state.value for i in self.simulator.platform.hosts ])
            self.simulator.switch_off(hosts_to_turnoff)

    def add_host_to_list(self, host: Host):
        if host.id not in self.idle_servers:
            self.idle_servers[host.id] = self.simulator.current_time
            t = self.simulator.current_time + self.timeout
            self.simulator.set_callback(t, self.shutdown_idle_hosts)

    def _on_host_state_changed(self, host: Host):
        if host.is_idle:
            if host.id not in self.idle_servers:
                self.add_host_to_list(host)
            else:
                self.idle_servers.pop(host.id, None)

    def _on_sim_begins(self, _):
        self.idle_servers.clear()
        for h in self.simulator.platform.hosts:
            if h.is_idle:
                self.add_host_to_list(h)


