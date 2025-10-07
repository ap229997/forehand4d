class DatasetRegistry:
    _datasets = {}

    @classmethod
    def register(cls, name):
        def wrapper(dataset_class):
            if isinstance(name, str):
                cls._datasets[name] = dataset_class
            elif isinstance(name, (list, tuple)):
                for n in name:
                    cls._datasets[n] = dataset_class
            return dataset_class
        return wrapper

    @classmethod
    def get_dataset(cls, name):
        if name not in cls._datasets:
            raise KeyError(f"Dataset {name} not registered")
        return cls._datasets[name]
