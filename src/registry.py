from typing import Type, Dict, Any

class ModelRegistry:
    _models: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register model classes"""
        def wrapper(model_class):
            if isinstance(name, str):
                cls._models[name] = model_class
            elif isinstance(name, (list, tuple)):
                for n in name:
                    cls._models[n] = model_class
            return model_class
        return wrapper

    @classmethod
    def get_model(cls, name: str):
        """Get model class by name"""
        if name not in cls._models:
            raise KeyError(f"Model {name} not registered")
        return cls._models[name]