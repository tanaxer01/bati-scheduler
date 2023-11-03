from batsim_py import SimulatorHandler, SimulatorEvent, HostEvent
from batsim_py.resources import Host

from typing import Any, Optional, Tuple, Dict
import xml.etree.ElementTree as ET
from gridgym.envs.grid_env import GridEnv
from gym import error, spaces


class DspEnv(GridEnv):
    def __init__(self,
                 platform_fn: str, workloads_dir: str,
                 t_action: int=1, t_shutdown: int=0,
                 hosts_per_server: int=1, queue_max_len: int=20, seed: Optional[int] = None,
                 external_events_fn: Optional[str]   = None,
                 simulation_time:    Optional[float] = None) -> None:

        self.queue_max_len = queue_max_len
        self.t_action = t_action

        super().__init__(platform_fn, workloads_dir, seed, external_events_fn,
                         simulation_time, True, hosts_per_server=hosts_per_server)

        self.shutdown_policy = ShutdownPolicy(t_shutdown, self.simulator)
        #self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self._on_simulation_begins)

        root = ET.parse(platform_fn).getroot()

        prefixes = { "G": 10e9, "M": 10e6, "K": 10e3 }
        to_int = lambda x: float(x[:-1]) * prefixes[x[-1]]

        self.host_speeds = { h.attrib["id"]: to_int(h.get("speed").split(",")[0][:-1])
                                for h in root.iter("host") }

    def _on_simulation_begins(self, _):
        for i in self.simulator.platform.hosts:
            print(i.id, i.name)
        pass

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        return ({}, 0, False, {})

    def _get_reward(self) -> float:


        # aprox. turnaround time
        aprox_turnaround = 0.
        print(self.simulator.agenda)



        return 0.

    def _get_state(self) -> Any:
        return ""

    def _get_spaces(self) -> Tuple[spaces.Dict, spaces.Discrete]:
        return spaces.Dict(), spaces.Discrete(1)

class ShutdownPolicy():
    def __init__(self, timeout: int, simulator: SimulatorHandler):
        super().__init__()
        self.timeout = timeout
        self.simulator = simulator
        self.idle_servers: Dict[int, float] = {}

        self.simulator.subscribe(
            HostEvent.STATE_CHANGED, self._on_host_state_changed)
        self.simulator.subscribe(
            SimulatorEvent.SIMULATION_BEGINS, self._on_sim_begins)

    def shutdown_idle_hosts(self, *args, **kwargs):
        hosts_to_turnoff = []
        for h_id, start_t in list(self.idle_servers.items()):
            if self.simulator.current_time - start_t >= self.timeout:
                hosts_to_turnoff.append(h_id)
                del self.idle_servers[h_id]

        if hosts_to_turnoff:
            self.simulator.switch_off(hosts_to_turnoff)

    def _on_host_state_changed(self, host: Host):
        if host.is_idle:
            if host.id not in self.idle_servers:
                self.idle_servers[host.id] = self.simulator.current_time
                t = self.simulator.current_time + self.timeout
                self.simulator.set_callback(t, self.shutdown_idle_hosts)
        else:
            self.idle_servers.pop(host.id, None)

    def _on_sim_begins(self, _):
        self.idle_servers.clear()
        for h in self.simulator.platform.hosts:
            if h.is_idle:
                self.idle_servers[h.id] = self.simulator.current_time
                t = self.simulator.current_time + self.timeout
                self.simulator.set_callback(t, self.shutdown_idle_hosts)

