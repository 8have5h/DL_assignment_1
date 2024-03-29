
def get_normalization(dim , type):
    if type == "torch_bn":
        return nn.BatchNorm2d(dim)
    elif type =="in":
        return InstanceNorm(dim)
    elif type == "ln":
        return LayerNorm(dim)
    elif type == "bn":
        return BatchNorm(dim)
    elif type == "gn":
        return GroupNorm(dim)
    elif type == "bin":
        return BatchInstanceNorm(dim)
    elif type == "nn":
        return NoNorm()
    else:
        raise ValueError("Invalid normalization type" + type)