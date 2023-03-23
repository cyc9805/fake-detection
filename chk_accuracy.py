from model import ft_zi2zi_Gen
import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
import seaborn
import matplotlib.pyplot as plt


def main(args):

    # load data
    size = 80
    resize = (size, )
    transform_list = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), (1, 1, 1)),
            transforms.Normalize(0.5, 0.5),
        ])

    mod_data_dir = os.path.join(args.current_dir, 'data', 'modified')
    refer_data_dir = os.path.join(args.current_dir, 'data', 'reference')
    orig_data_dir = os.path.join(args.current_dir, 'data', 'original')

    hospital_mod_list = os.listdir(mod_data_dir)
    hospital_refer_list = os.listdir(refer_data_dir)
    hospital_orig_list = os.listdir(orig_data_dir)

    hospital_mod_dict = {}
    hospital_refer_dict = {}
    hospital_orig_dict = {}

    for mod_folder in hospital_mod_list:
        if mod_folder == '.DS_Store':
            continue
        mod_imgs = datasets.ImageFolder(os.path.join(mod_data_dir, mod_folder), transform_list)
        dataloader = torch.utils.data.DataLoader(mod_imgs, batch_size=1, shuffle=False)
        hospital_mod_dict[mod_folder] = dataloader

    for refer_folder in hospital_refer_list:
        if refer_folder == '.DS_Store':
            continue
        refer_imgs = datasets.ImageFolder(os.path.join(refer_data_dir, refer_folder), transform_list)
        dataloader = torch.utils.data.DataLoader(refer_imgs, batch_size=1, shuffle=False)
        hospital_refer_dict[refer_folder] = dataloader

    for orig_folder in hospital_orig_list:
        if orig_folder == '.DS_Store':
            continue
        orig_imgs = datasets.ImageFolder(os.path.join(orig_data_dir, orig_folder), transform_list)
        dataloader = torch.utils.data.DataLoader(orig_imgs, batch_size=1, shuffle=False)
        hospital_orig_dict[orig_folder] = dataloader

    pt_model = ft_zi2zi_Gen()
    load_filename = 'pt_zi2zi_gen.pth'
    load_path = os.path.join(args.current_dir, load_filename)
    pt_model.load_state_dict(torch.load(load_path))
    pt_model.train(False)


    total_neg_hit, total_pos_hit = 0, 0
    total_neg_num, total_pos_num = 0, 0
    total_neg_miss, total_pos_miss = 0, 0
    total_f1_score = 0

    for hospital in sorted(list(hospital_mod_dict)):

        mod_features = torch.tensor([])
        mod_labels = torch.tensor([])

        for mod_image, mod_label in  hospital_mod_dict[hospital]:
            out_feature = pt_model(mod_image, set_classifier=False)
            mod_features = torch.concat((mod_features, out_feature), dim=0)
            mod_labels = torch.concat((mod_labels, mod_label), dim=0)

        refer_features = torch.tensor([])
        refer_labels = torch.tensor([])

        for refer_image, refer_label in hospital_refer_dict[hospital]:
            out_feature = pt_model(refer_image, set_classifier=False)
            refer_features = torch.concat((refer_features, out_feature), dim=0)
            refer_labels = torch.concat((refer_labels, refer_label), dim=0)

        orig_features = torch.tensor([])
        orig_labels = torch.tensor([])

        for orig_image, orig_label in hospital_refer_dict[hospital]:
            out_feature = pt_model(orig_image, set_classifier=False)
            orig_features = torch.concat((orig_features, out_feature), dim=0)
            orig_labels = torch.concat((orig_labels, orig_label), dim=0)

        mod_norm = np.linalg.norm(mod_features, axis=1)
        ref_norm = np.linalg.norm(refer_features, axis=1)
        orig_norm = np.linalg.norm(orig_features, axis=1)

        mod_norm = np.expand_dims(mod_norm, axis=1)
        ref_norm = np.expand_dims(ref_norm, axis=1)
        orig_norm = np.expand_dims(orig_norm, axis=1)

        mod_features /= mod_norm
        refer_features /= ref_norm
        orig_features /= orig_norm

        sim_matrix_mod = np.dot(mod_features, refer_features.T)
        sim_matrix_orig = np.dot(orig_features, refer_features.T)

        ax_mod = seaborn.heatmap(sim_matrix_mod)
        ax_orig = seaborn.heatmap(sim_matrix_orig)

        plt.figure()
        fig_mod = ax_mod.get_figure()
        fig_mod.savefig("sim_maps/modified_" + hospital + '.png')
        plt.clf()

        plt.figure()
        fig_orig = ax_orig.get_figure()
        fig_orig.savefig("sim_maps/original_" + hospital + '.png')
        plt.clf()

        print(f'Similarity map for {hospital} is created!')

        sim_mean_mod = np.mean(sim_matrix_mod, 1)
        sim_mean_orig = np.mean(sim_matrix_orig, 1)

        neg_hit = 0.0
        neg_miss = 0.0
        for mean in sim_mean_mod:
            if mean * 100 < args.threshold:
                neg_hit += 1
            else:
                neg_miss += 1

        tn_rate = neg_hit / len(sim_mean_mod)
        total_neg_hit += neg_hit
        total_neg_miss += neg_miss
        total_neg_num += len(sim_mean_mod)
        print('True Negative rate for {} is {:.2f}'.format(hospital, tn_rate))

        pos_hit = 0.0
        pos_miss = 0.0
        for mean in sim_mean_orig:
            if mean * 100 >= args.threshold:
                pos_hit += 1
            else:
                pos_miss += 1

        recall = pos_hit / len(sim_mean_orig)
        total_pos_hit += pos_hit
        total_pos_miss += pos_miss
        total_pos_num += len(sim_mean_orig)
        
        try:
            precision = pos_hit / (pos_hit + neg_miss)
        except:
            precision = pos_hit / 1e-6
        
        try:
            f1_score = 2*precision*recall / (precision+recall)
        except:
            f1_score = 2*precision*recall / 1e-6

        acc = (pos_hit + neg_hit) / (len(sim_mean_mod) + len(sim_mean_orig))

        print('Recall for {} is {:.2f}'.format(hospital, recall))
        print('Accuracy for {} is {:.2f}'.format(hospital, acc))
        print('F1 score for {} is {:.2f}'.format(hospital, f1_score))
        print('')

    total_recall = total_pos_hit/total_pos_num

    try:
        total_precision = total_pos_hit / (total_pos_hit + total_neg_miss)
    except:
        total_precision = total_pos_hit / 1e-6
    
    try:
        total_f1_score = 2*total_precision*total_recall / (total_precision+total_recall)
    except:
        total_f1_score = 2*total_precision*total_recall / 1e-6

    print('Overall True Negative rate is {:.2f}'.format(total_neg_hit/total_neg_num))
    print('Overall Recall is {:.2f}'.format(total_recall))
    print('Overall Accuracy is {:.2f}'.format((total_pos_hit + total_neg_hit)/ (total_pos_num + total_neg_num)))
    print('Overall F1 score is {:.2f}'.format(total_f1_score))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer_similiarity_map')
    parser.add_argument('--current_dir', default='/home/cyc/fake_detection', type=str, help='Path to the current directory')
    parser.add_argument('--pt_zi2zi_name', type=str, default='pt_zi2zi_gen.pth', help='Name of the model should end with .pth format')
    parser.add_argument('--threshold', type=float, default=92, help='Threshold for accuracy')
    args = parser.parse_args()
    with torch.no_grad():
        main(args)

