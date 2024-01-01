from model_loader import Models
from data_selector import DataSelector
from data_generateor import DataGenerator

selector_config_path = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
selector_checkpoint_path = 'pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth'
target_config_path = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
target_checkpoint_path = 'pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth'
original_data_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/val'
generated_data_path = '../data/AI_test/'
label_path = '../data/gtFine_trainvaltest/gtFine/val/'
selected_data_path = '../data/selected/'


# generate mutations on dataset
def generate():
    generator = DataGenerator(data_path=original_data_path, out_path=generated_data_path, sample_per_folder=30)
    generator.generate()


# select mutations with the selector models. to add more model, use models.append_model
def select():
    models = Models(data_path=generated_data_path, label_path=label_path)
    models.append_model(selector_config_path, selector_checkpoint_path)
    selector = DataSelector(generated_data_path, label_path, selected_data_path, top_k=3, use_increment=False)
    selector.select(models)


# test on the target models. to add more model, use models.append_model
def model_test():
    models = Models(data_path=selected_data_path, label_path=label_path)
    models.append_model(target_config_path, target_checkpoint_path)
    models.traverse(top_k=3)


def main():
    # generate()
    # select()
    model_test()


if __name__ == "__main__":
    main()
