from . import sliceT, traceT


class TransformWrapper:
    def __init__(self, func, **params):
        self.func = func
        self.params = params

    def __call__(self, tensor):
        return self.func(tensor, **self.params)
    
    def __name__(self):
        return self.func.__name__

def get_transform(transform_type, transform_name, params):
    if transform_type == 'slice':
        if transform_name in dir(sliceT):
            return TransformWrapper(getattr(sliceT, transform_name), **params)
    elif transform_type == 'trace':
        if transform_name in dir(traceT):
            return TransformWrapper(getattr(traceT, transform_name), **params)
    raise ValueError(f"Undefined transformation: {transform_name} for type {transform_type}")

def build_transforms(config):
    transform_type = config['type']
    transforms = []
    for transform_name, params in config['transformations']:
        transforms.append(get_transform(transform_type, transform_name, params))
    return transforms
