import yaml
import datetime

class Config:
    def __init__(self):
        # load default config file
        with open('kge/config-default.yaml', 'r') as file:
            self.options = yaml.load(file)

    def load(self, filename):
        # user_config = yaml.load(file)
        raise NotImplementedError

    def folder(self):
        return self.options['output']['folder']

    def logfile(self):
        return self.folder() + "/" + self.options['output']['logfile']

    def log(self, msg, echo=True):
        if echo:
            print(msg)
        with open(self.logfile(), 'a') as file:
            for line in msg.splitlines():
                file.write(str(datetime.datetime.now()))
                file.write(" ")
                file.write(line)
                file.write("\n")

    def dump(self, filename):
        with open(filename, "w+") as file:
            file.write( yaml.dump(self.options) )

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
        for i in range(len(splits)-1):
            data = data[splits[i]]

        current_value = data.get(splits[-1])
        if current_value is None:
            if not create:
                raise ValueError("field {} not present".format(field))
        elif type(value) != type(current_value):
            raise ValueError("field {} has incorrect type".format(field))

        data[splits[-1]] = value
        return value
