model_defaults = {
    'densenet121': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'wideresnet50': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet18': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet34': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet50': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'resnet101': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'resnet18_ms': {
        'target_resolution': (224, 224),
    },
    'convnet': {
        'target_resolution': (224, 224),
    },
    'vit': {
        'model_kwargs': {
            'model_size': 'B_16',
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
}
