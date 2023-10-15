from typing import Iterable, List, Set
from itertools import combinations
import numpy as np

class FutureReservation:
    def __init__(self, host: int, start: float, end: float) -> None:
        self.host = host
        self.start, self.end = start, end

    def __repr__(self) -> str:
        return f"<< FutureReservation [h{self.host}] {self.start} - {self.end} >>"

class FreeSpace:
    def __init__(self, hosts: Set[int], start: float, end: float) -> None:
        assert len(hosts) != 0

        self.hosts = hosts
        self.start, self.end = start, end

    def __repr__(self) -> str:
        return f"<< FreeSpace [h{self.hosts}] {self.start} - {self.end} >>"


class JobAgenda:
    def __init__(self, total_resources: int, max_range: float):
        self.total_resources = total_resources
        self.max_range, self.curr_time = max_range, 0.

        self.items : List[FutureReservation] = []

    @property
    def max_time(self) -> float:
        return self.curr_time + self.max_range

    def _check_range(self, start: float, end: float) -> Iterable[FutureReservation]:
        return filter(lambda x: start <= x.start < end or start < x.end <= end, self.items)

    def add_reservation(self, hosts_ids: list[int], start: float, end: float) -> None:
        assert start <= self.max_time, "Reservation starts after max time."
        for host_id in hosts_ids:
            assert host_id not in [ i.host for i in self._check_range(start, end) ], "Host already allocated in that time range."
            self.items.append( FutureReservation(host_id, start, end) )

    def update(self, current_time):
        self.curr_time = current_time
        self.items = list(filter(lambda x: x.end > current_time, self.items))

        for i in self.items:
            i.start = max(current_time, i.start)

    def get_valid_spaces(self, estimated_duration: float, needed_cores: int) -> List[FreeSpace]:
        """
        """
        free_spaces = []

        # TODO: Check if necesary to allow diferent steps in this for. Migth be
        #       usefull in cases where diferences lower than 1. happen a lot.

        for i in np.arange(self.curr_time, self.max_time+1., 1.):
            start, end = i, i + estimated_duration
            used_cores = [ i.host for i in self._check_range(start, end) ]

            if self.total_resources - len(used_cores) >= needed_cores:
                free_cores = { i for i in range(self.total_resources) if i not in used_cores }

                if len(free_cores) > needed_cores:
                    for cores in combinations(free_cores, needed_cores):
                        free_spaces.append(FreeSpace(set(cores), start, end))
                else:
                    free_spaces.append(FreeSpace(free_cores, start, end))

        return free_spaces

    def print_agenda(self):
        hosts = [ [ 0 for _ in range(int(self.max_range)+1)] for _ in range(self.total_resources) ]

        zero = self.curr_time
        for i in self.items:
            for j in range(int(i.start - zero), int(i.end - zero + 1)):
                hosts[i.host-1][j] = i.host

        for i in hosts:
            print(i)
        print()

if __name__ == "__main__":

    agenda = [
        (1, 0., 4.),
        (2, 0., 3.),
        (4, 1., 4.),
        (2, 10., 11.)
    ]


    listFreeSpaces = JobAgenda(5, 15)

    for i, j, k in agenda:
        listFreeSpaces.add_reservation([i], j, k)
    listFreeSpaces.add_reservation([4,5], 6., 8.)

    listFreeSpaces.print_agenda()
    print("===")
    spaces = listFreeSpaces.get_valid_spaces(4., 2)
    print(len(spaces))
    for i in spaces[:3]:
        print(i)
    print("===")
    listFreeSpaces.update(4.)
    listFreeSpaces.print_agenda()

