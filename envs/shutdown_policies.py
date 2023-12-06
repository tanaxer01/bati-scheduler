
from typing import Dict
from batsim_py import HostEvent, SimulatorEvent, SimulatorHandler
from batsim_py.resources import Host, PowerStateType

class ShutdownPolicy:
    pass


class ShutdownOnAdvance:
    pass



class TimeoutPolicy:
    def __init__(self, t_timeout: float, simulator: SimulatorHandler) -> None:
        self.simulator = simulator
        self.t_timeout = t_timeout
        self.hosts_idle = {}
        # Subscribe to some events.
        self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self.on_simulation_begins)
        self.simulator.subscribe(HostEvent.STATE_CHANGED, self.on_host_state_changed)

    def on_simulation_begins(self, s: SimulatorHandler) -> None:
        for host in s.platform.hosts:
            if host.is_idle:
                self.hosts_idle[host.id] = s.current_time
                self.setup_callback()

    def on_host_state_changed(self, h: Host) -> None:
        print(self.hosts_idle)
        print(h.id, h.is_idle, not h.is_allocated)

        if (h.is_idle and not h.is_allocated) and not h.id in self.hosts_idle:
            self.hosts_idle[h.id] = self.simulator.current_time
            self.setup_callback()
        elif (not h.is_idle or h.is_allocated) and h.id in self.hosts_idle:
            del self.hosts_idle[h.id]

    def setup_callback(self) -> None:
        t_next_call = self.simulator.current_time + self.t_timeout
        self.simulator.set_callback(t_next_call, self.callback)

    def callback(self, current_time: float) -> None:
        for host_id, t_idle_start in list(self.hosts_idle.items()):
            host = self.simulator.platform.get_host(host_id)
            if  current_time - t_idle_start >= self.t_timeout and (host.is_idle and not host.is_allocated):
                print("OFF", host.id, host.is_idle, not host.is_allocated)
                self.simulator.switch_off([host_id])
