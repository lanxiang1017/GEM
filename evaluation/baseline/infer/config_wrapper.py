# infer/config_wrapper.py
import yaml

class ConfigWrapper:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key):
        return self.config.get(key)
    
    def get_id(self, data):
        if isinstance(self.config.get('id_key'), str):
            return data.get(self.config.get('id_key'), None)
        elif isinstance(self.config.get('id_key'), list):
            return '_'.join([str(data.get(key, None)) for key in self.config.get('id_key')])
