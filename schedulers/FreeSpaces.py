from typing import List, Set
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

class FreeSpaceContainer:
    def __init__(self, total_resources: int, max_time: float):
        self.agenda : List[FutureReservation] = []
        self.total_resources = total_resources
        self.max_time = max_time

    def add(self, host_id: int, start: float, end: float) -> None:
        # Check if a conflix exists
        cores_found = filter(lambda x: start <= x.start < end or start < x.end <= end, self.agenda)
        assert host_id not in [ i.host for i in cores_found ], "Host is already allocated in this time range."

        self.agenda.append( FutureReservation(host_id, start, end) )

    def update(self, current_time):
        self.agenda = list(filter(lambda x: x.end < current_time, self.agenda))

    def get_spaces(self, task_duration: float, cores_needed: int) -> List[FreeSpace]:
        free_spaces = []
        for i in np.arange(0., self.max_time+1., 1.):
            start, end = i, i+task_duration
            used_cores = list(filter(lambda x: start <= x.start < end or start < x.end <= end, self.agenda))

            if self.total_resources - len(used_cores) >= cores_needed:
                free_cores = { i for i in range(self.total_resources) if i not in [ j.host for j in used_cores ] }
                free_spaces.append(FreeSpace(free_cores, start, end))

        return free_spaces

if __name__ == "__main__":
    def print_agenda(N, T, agenda):
        hosts = [ [ 0 for _ in range(T)] for _ in range(N) ]

        for i in agenda:
            for j in range(int(i.start), int(i.end+1)):
                hosts[i.host-1][j] = i.host

        for i in hosts:
            print(i)
        print()

    agenda = [
        (0, 0., 4.),
        (1, 0., 3.),
        (3, 1., 4.),
        (3, 5., 8.),
        (4, 5., 8.),
        (1, 9., 11.)
    ]

    listFreeSpaces = FreeSpaceContainer(20, 5)
    for i, j, k in agenda:
        listFreeSpaces.add(i, j, k)

    print_agenda(5, 12, listFreeSpaces.agenda)

    spaces = listFreeSpaces.get_spaces(4., 3)
    for i in spaces:
        print( i.hosts,"|", i.start, i.end )

    listFreeSpaces.update(8.)

    spaces = listFreeSpaces.get_spaces(4., 3)
    for i in spaces:
        print( i.hosts,"|", i.start, i.end )

