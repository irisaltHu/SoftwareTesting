from mmseg.apis import init_model, inference_model
import matplotlib.pyplot as plt
import os


def main():
    config_path = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_path = 'pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth'
    model = init_model(config_path, checkpoint_path)

    data_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/val/'
    folder_names = os.listdir(data_path)
    for folder_name in folder_names:
        file_path = data_path + folder_name
        filenames = os.listdir(file_path)
        batch_size = 4
        for i in range(0, int(len(filenames) / batch_size), batch_size):
            img_paths = filenames[i * batch_size:(i + 1) * batch_size]
            img_paths = [file_path + '/' + img for img in img_paths]
            results = inference_model(model, img_paths)
            for result in results:
                img = result.pred_sem_seg.get('data').cpu()
                img = img.numpy()[0]
                plt.imshow(img)
                plt.show()


if __name__ == "__main__":
    main()
