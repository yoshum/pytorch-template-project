from .linear import LinearModel


def build_net(config):
    import torchvision.models as models

    if config is None:
        return LinearModel()

    if isinstance(config, str):
        return models.__dict__[config]
    return models.__dict__[config.type](**config.get("parameters", {}))
