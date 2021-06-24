from collections import defaultdict
from typing import Dict, List
from stopwatch import Stopwatch


class TimerCollection:

    _timers = defaultdict(Stopwatch)

    def __getitem__(self, timer_name: str) -> Stopwatch:
        return self._timers[timer_name]

    def extend_all(self, other: "TimerCollection") -> None:
        """Adds all laps of the `other`'s timers to the corresponding timers of this TimerCollection.

        """
        for name in other.keys():
            self[name]._laps.extend(other[name]._laps)

    def reset(self, timer_names: List[str] = None) -> None:
        """Resets timers with the names in `timer_names`. If None is given, all timers are reset.

        """
        names = timer_names or self.keys()
        for name in names:
            self[name].reset()

    def elapsed(self, timer_names: List[str] = None) -> Dict[str, float]:
        """Calculates and returns a dictionary with the elapsed times of the timers given in `timer_names`. If None is
        given, times for all timers are returned.

        """
        names = timer_names or self.keys()
        return {name: self[name].elapsed for name in names}

    def keys(self):
        return self._timers.keys()
