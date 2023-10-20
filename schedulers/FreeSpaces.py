from typing import Iterable, Iterator, List, Set
from itertools import combinations
import random

class TimeBlock:
    def __init__(self, start: int, end: int) -> None:
        self.start, self.end = start, end

    def __repr__(self) -> str:
        return f"{self.start} -> {self.end}"

class FutureReservation(TimeBlock):
    def __init__(self, hosts: Set[int], job_id: str, start: int, end: int) -> None:
        super().__init__(start, end)
        self.hosts = hosts
        self.job   = job_id
        self.full  = False
        self.started = False

    def __repr__(self) -> str:
        return f"<< FutureReservation [{[ 'h'+str(i) for i in self.hosts ]}] {'j'+str(self.job)} {self.start} -> {self.end} >>"

    def add_host(self, host_id) -> None:
        assert host_id not in self.hosts
        self.hosts.add(host_id)

class FreeSpace(TimeBlock):
    def __init__(self, hosts: Set[int], start: int, end: int) -> None:
        super().__init__(start, end)

        assert len(hosts) != 0
        self.hosts = hosts

    def __repr__(self) -> str:
        return f"<< FreeSpace [h{self.hosts}] {self.start} - {self.end} >>"


class JobAgenda:
    def __init__(self, total_resources: int, max_range: int, min_step: int):
        self.total_resources = total_resources
        self.max_range  = max_range
        self.min_step   = min_step
        self.curr_time  = 0

        self.items : List[FutureReservation] = []

    @property
    def max_time(self) -> int:
        return self.curr_time + self.max_range

    @property
    def jobs(self) -> Iterator[str]:
        return ( i.job for i in self.items )

    @property
    def running_jobs(self) -> Iterator[str]:
        return ( i.job for i in self.items if i.started )

    def generator(self) -> Iterator[FutureReservation]:
        return ( i for i in self.items )

    def _check_range(self, start: float, end: float) -> Iterable[FutureReservation]:
        smaller = filter(lambda x:   start <= x.start <=   end or   start <= x.end <=   end, self.items)
        bigger  = filter(lambda x: x.start <=   start <= x.end or x.start <=   end <= x.end, self.items)

        items = set(smaller).union(set(bigger))

        return items

    def add_reservation(self, hosts: Set[int], job_id: str, start: int, end: int) -> None:
        assert start <= self.max_time, "Reservation starts after max time."
        assert sum([ len(hosts & i.hosts) for i in self._check_range(start, end) ]) == 0, "A host is already allocated in this time range."

        item = FutureReservation(hosts, job_id, start, end)
        item.full = True
        self.items.append(item)

    def update_reservation(self, host_id: int, job_id: int, start, end) -> None:
        reservations = list(filter(lambda x: x.job == job_id, self._check_range(start, end)))
        assert len(reservations) == 1, "Job cant be reserved more than one time in this time period."

        reservations[0].add_host(host_id)

    def start_job(self, job_id: str) -> None:
        for i in self.generator():
            if i.start == self.curr_time and i.job == job_id and not i.started:
                i.started = True
                return

        raise Exception("job running or not found")

    def update_time(self, current_time: int):
        self.curr_time = current_time
        self.items = list(filter(lambda x: x.end >= current_time, self.items))

        for i in self.items:
            if i.start < self.curr_time:
                assert i.started, f"If job start is {i.start} it should had already started at this point (curr_time = {self.curr_time})"
            i.start = max(current_time, i.start)

    def get_posible_spaces(self, estimated_duration: int, needed_cores: int) -> List[FreeSpace]:
        free_spaces = []

        for i in range(self.curr_time, self.max_time + 1, self.min_step):
            start, end = i, i + estimated_duration
            used_cores = set().union(*[ i.hosts for i in self._check_range(start, end) ])

            if self.total_resources - len(used_cores) >= needed_cores:
                free_cores = { i for i in range(self.total_resources) if i not in used_cores }

                # TODO check if still necesary
                #if len(free_cores) > needed_cores:
                #    for cores in combinations(free_cores, needed_cores):
                #        free_spaces.append(FreeSpace(set(cores), start, end))
                #else:
                #    free_spaces.append(FreeSpace(free_cores, start, end))
                free_spaces.append(FreeSpace(free_cores, start, end))

        return free_spaces

    def print_agenda(self):
        hosts = [ [ 0 for _ in range(int(self.max_range)+1)] for _ in range(self.total_resources) ]

        zero = self.curr_time
        for i in self.items:
            for k in range(int(i.start - zero), int(i.end - zero + 1)):
                for j in i.hosts:
                    hosts[j-1][k] = j

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

    listFreeSpaces = JobAgenda(5, 20, 1)

    for i, j, k in agenda:
        listFreeSpaces.add_reservation({i}, "0", j, k)
    listFreeSpaces.add_reservation({4,5}, "0", 6, 8)

    listFreeSpaces.print_agenda()
    print("===")
    spaces = listFreeSpaces.get_posible_spaces(4, 2)
    print(len(spaces))
    for i in spaces[:3]:
        print(i)
    print("===")
    listFreeSpaces.add_reservation({1,2}, "4",3, 2)

