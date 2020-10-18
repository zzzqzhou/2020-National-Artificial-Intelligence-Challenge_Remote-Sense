from models.deeplabv3plus import DeepLabV3Plus


def get_model(model, backbone, pretrained, nclass):
    if model == "deeplabv3plus":
        model = DeepLabV3Plus(backbone, pretrained, nclass)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    params_num = sum(p.numel() for p in model.parameters())
    print("\nParams: %.1fM" % (params_num / 1e6))

    return model