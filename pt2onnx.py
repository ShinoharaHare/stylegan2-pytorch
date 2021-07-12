import argparse
import torch

from os.path import join, basename, splitext
from model import Generator

def main():
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('path', metavar='.PT')

    args = parser.parse_args()

    G = Generator(args.size, args.latent, args.n_mlp, args.channel_multiplier).to(device)
    ckpt = torch.load(args.path)
    state_dict = {'truncation_latent': ckpt['latent_avg']}
    state_dict.update(ckpt['g_ema'])
    G.load_state_dict(state_dict)

    path = join(args.outdir, splitext(basename(args.path))[0] + '.onnx')
    latents = torch.rand(1, G.num_layers + 1, 512)
    truncation = torch.tensor(0.5)

    torch.onnx.export(
        model=G, 
        args=(latents, truncation),
        export_params=True,
        f=path,
        verbose=False,
        training=False,
        do_constant_folding=True,
        input_names=['latents', 'truncation'],
        output_names=['images'],
        opset_version=10
    )


if __name__ == '__main__':
    main()
    