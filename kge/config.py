import yaml
import datetime
import time
import os

class Config:
    """Configuration options."""

    def __init__(self):
        # load default config file
        with open('kge/config-default.yaml', 'r') as file:
            self.options = yaml.load(file, Loader=yaml.SafeLoader)

    def load(self, filename, create=False):
        """Update configuration options from the specified YAML file."""
        with open(filename, 'r') as file:
            new_options = yaml.load(file, Loader=yaml.SafeLoader)
        self.set_all(new_options, create)

    def folder(self):
        return self.get('output.folder')

    def logfile(self):
        return self.folder() + "/" + self.get('output.logfile')

    def tracefile(self):
        return self.folder() + "/" + self.get('output.tracefile')

    def checkpointfile(self, epoch):
        "Return path of checkpoint file for given epoch"
        return "{}/{}_{:05d}.pt".format(
            self.folder(), self.get('checkpoint.basefile'), epoch)

    def last_checkpointfile(self):
        # find last checkpoint file (stupid but works)
        tried_epoch = 0
        found_epoch = 0
        while tried_epoch < found_epoch + 100:
            tried_epoch += 1
            if os.path.exists(self.checkpointfile(tried_epoch)):
                found_epoch = tried_epoch
        if found_epoch>0:
            return self.checkpointfile(found_epoch)
        else:
            return None


    def log(self, msg, echo=True, prefix=''):
        """Add an entry to the default log file. Optionally also print on console."""
        with open(self.logfile(), 'a') as file:
            for line in msg.splitlines():
                if prefix:
                    line = prefix + line
                if echo:
                    print(line)
                file.write(str(datetime.datetime.now()))
                file.write(" ")
                file.write(line)
                file.write("\n")

    def trace(self, echo=False, echo_prefix='', echo_flow=False,
              log=False, **kwargs):
        """Write a set of key value pairs to the trace file (in single YAML line).
        Optionally, also print on console and/or write in log file."""
        with open(self.tracefile(), 'a') as file:
            kwargs['timestamp']=time.time()
            line = yaml.dump(kwargs, width=float('inf'), default_flow_style=True).strip()
            if echo or log:
                msg = yaml.dump(kwargs, default_flow_style=echo_flow)
                if log:
                    self.log(msg, echo, echo_prefix)
                else:
                    for line in msg.splitlines():
                        if prefix:
                            line = prefix + line
                            print(line)
            file.write(line)
            file.write("\n")
            return line

    def dump(self, filename):
        """Dump this dictionary to the given file."""
        with open(filename, "w+") as file:
            file.write(yaml.dump(self.options))

    def get(self, field):
        """Obtain value of specified field. Nested dictionaries can be accessed via "."""
        result = self.options
        for name in field.split('.'):
            result = result[name]
        return result

    def set(self, field, value, create=False):
        """Set value of specified field. Nested dictionaries can be accessed via "."""
        splits = field.split('.')
        data = self.options
        for i in range(len(splits) - 1):
            data = data[splits[i]]

        current_value = data.get(splits[-1])
        if current_value is None:
            if not create:
                raise ValueError("field {} not present".format(field))
        elif type(value) != type(current_value):
            raise ValueError("field {} has incorrect type".format(field))

        data[splits[-1]] = value
        return value

    def flatten(options):
        result = {}
        Config.__flatten(options, result)
        return result

    def __flatten(options, result, prefix=''):
        for key, value in options.items():
            fullkey = key if prefix == '' else prefix + '.' + key
            if type(value) is dict:
                Config.__flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value

    def set_all(self, new_options, create=False):
        for key, value in Config.flatten(new_options).items():
            self.set(key, value, create)

    def check(self, field, values):
        if not self.get(field) in values:
           raise ValueError(field)
