import importlib

class ModelLoader:
    def __init__(self, model_name, config, use_accel):
        self.model_name = model_name
        self.config = config
        self._model = None
        self.use_accel = use_accel

    def _lazy_import(self, module_name, func_name):
        """Dynamically import a module and return the desired function."""
        if module_name.startswith('.'):
            # Convert relative import to absolute import based on the current package context
            module_name = __package__ + module_name
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    @property
    def model(self):
        """Load and return the model instance, if not already loaded."""
        if self._model is None:
            load_func = self._lazy_import(self.config['load'][0], self.config['load'][1])
            if self.config.get('call_type') == 'api':
                self._model = load_func(
                    self.config['model_path_or_name'], 
                    self.config['base_url'], 
                    self.config['api_key'], 
                    self.config['model']
                )
            else:
                self._model = load_func(self.model_name, self.config, use_accel=self.use_accel)
        return self._model

    @property
    def infer(self):
        """Return the inference function."""
        return self._lazy_import(self.config['infer'][0], self.config['infer'][1])

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config, use_accel):
        """Register a model configuration."""
        self.models[name] = ModelLoader(name, config, use_accel)

    def load_model(self, choice, use_accel=False):
        """Load a model based on the choice."""
        if choice in self.models:
            return self.models[choice].model
        else:
            raise ValueError(f"Model choice '{choice}' is not supported.")

    def infer(self, choice):
        """Get the inference function for a given model."""
        if choice in self.models:
            return self.models[choice].infer
        else:
            raise ValueError(f"Inference choice '{choice}' is not supported.")

# Initialize model registry
model_registry = ModelRegistry()

# Configuration of models
model_configs = {
    'gpt4o': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4o',
        'base_url': None,
        'api_key': None,
        'model': 'gpt-4o',
        'call_type': 'api'
    },
    'Qwen2-VL-7B': {
        'load': ('.qwen2_vl_chat', 'load_model'),
        'infer': ('.qwen2_vl_chat', 'infer'),
        'model_path_or_name': '/path/to/Qwen2-VL-7B-Instruct',
        'call_type': 'local',
        'tp': 1
    }
}


def load_model(choice, use_accel=False):
    """Load a specific model based on the choice."""
    model_registry.register_model(choice, model_configs[choice], use_accel)
    return model_registry.load_model(choice, use_accel)

def infer(choice):
    """Get the inference function for a specific model."""
    return model_registry.infer(choice)

