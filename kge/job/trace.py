import yaml
import pandas as pd


class Trace:
    """Utility class for handling traces."""
    def __init__(self, tracefile=None):
        self.entries = []
        if tracefile:
            self.load(tracefile)

    def load(self, tracefile):
        with open(tracefile, 'r') as file:
            self.kv_pairs = []
            for line in file:
                entry = yaml.load(line, Loader=yaml.SafeLoader)
                self.entries.append(entry)

    def filter(self, filter_dict={}):
        def predicate(entry):
            for key, value in filter_dict.items():
                entry_value = entry.get(key)
                if not entry_value or value != entry_value:
                    return False
            return True
        return list(filter(predicate, self.entries))

    def to_dataframe(self, filter_dict={}):
        filtered_entries = self.filter(filter_dict)
        return pd.DataFrame(filtered_entries)
