from model_loader import Models


def main():
    config_path = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_path = 'pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth'
    data_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/val/'
    label_path = '../data/gtFine_trainvaltest/gtFine/val/'
    models = Models(data_path=data_path, label_path=label_path)
    models.append_model(config_path, checkpoint_path)
    models.append_model(config_path, checkpoint_path)
    models.traverse(batch_size=2)


if __name__ == "__main__":
    main()
