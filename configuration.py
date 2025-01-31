import os
import yaml

CONFIGURATION_DIR = 'config'
CONFIGURATION_FILE = 'config.yaml'
CONFIGURATION_PATH = os.path.join(CONFIGURATION_DIR, CONFIGURATION_FILE)

class YamlParser:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.params = self._load_yaml()

    def _load_yaml(self):
        with open(self.yaml_file, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def get_param(self, key, default=None):
        return self.params.get(key, default)

    def set_params_as_attributes(self):
        for key, value in self.params.items():
            setattr(self, key, value)

class YamlToDict:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.params = self._load_yaml()

    def _load_yaml(self):
        with open(self.yaml_file, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def get_params_as_dict(self):
        return self.params


def assert_configuration_availability(module, configuration):
    assert module is not None,\
        "Object is None!"
    assert configuration is not None,\
        f"Configuration passed to {module} is None!"
    assert module._CONFIG_NAME in configuration,\
        f"Configuration for module {module} is not defined!"


"""
    Wrapper over standard dictionary class that automatically loads parameters
"""
class Configuration:
    def __init__(self):
        parser = YamlToDict(CONFIGURATION_PATH)
        self._dict = parser.get_params_as_dict()
 
    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def update(self, other_dict):
        self._dict.update(other_dict)

    def __repr__(self):
        return repr(self._dict)