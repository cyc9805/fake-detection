from model import ft_zi2zi_Gen, Zi2ZiModel
import os
import argparse
import torch
import copy

parser = argparse.ArgumentParser(description='Create_pretrained_model')
parser.add_argument('--current_dir', default='/home/cyc/fake_detection', type=str, help='Path to the current directory')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--pt_zi2zi_name', type=str, default='pt_zi2zi_gen.pth', help='Name of the model should end with .pth format')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--input_nc', type=int, default=1)


def main(n_layer):
    args = parser.parse_args()
    root = args.current_dir
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")

    # Create Zi2ZiModel
    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False,
        n_layer=n_layer
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    temp = model.netG.model
    l_c = 0
    while True:
        l_c += 1
        temp.up = torch.nn.Sequential()
        if l_c > 6:
            temp.down[1].stride = (1, 1)
        if temp.innermost == True:
            break
        temp = temp.submodule

    target_model = model.netG

    ft_model = ft_zi2zi_Gen()
    ft_model.train(False)
    ft_model.layer1.down = copy.deepcopy(target_model.model.down)
    ft_model.layer2.down = copy.deepcopy(target_model.model.submodule.down)
    ft_model.layer3.down = copy.deepcopy(target_model.model.submodule.submodule.down)
    ft_model.layer4.down = copy.deepcopy(target_model.model.submodule.submodule.submodule.down)
    ft_model.layer5.down = copy.deepcopy(target_model.model.submodule.submodule.submodule.submodule.down)
    ft_model.layer6.down = copy.deepcopy(target_model.model.submodule.submodule.submodule.submodule.submodule.down)
    ft_model.layer7.down = copy.deepcopy(
        target_model.model.submodule.submodule.submodule.submodule.submodule.submodule.down)
    ft_model.layer8.down = copy.deepcopy(
        target_model.model.submodule.submodule.submodule.submodule.submodule.submodule.submodule.down)

    save_filename = args.ft_zi2zi_name
    save_path = os.path.join(root, save_filename)
    torch.save(ft_model.cpu().state_dict(), save_path)


if __name__ == '__main__':
    n_layer=8
    with torch.no_grad():
        main(n_layer)

