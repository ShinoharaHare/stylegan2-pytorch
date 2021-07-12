import argparse
import os
import sys
import pickle
import math

import torch
import numpy as np

from torchvision import utils
from os.path import join, basename, splitext

from model import Generator, Discriminator


def convert_modconv(vars, source_name, target_name, flip=False):
    weight = vars[source_name + '/weight'].value().eval()
    mod_weight = vars[source_name + '/mod_weight'].value().eval()
    mod_bias = vars[source_name + '/mod_bias'].value().eval()
    noise = vars[source_name + '/noise_strength'].value().eval()
    bias = vars[source_name + '/bias'].value().eval()

    dic = {
        'conv.weight': np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        'conv.modulation.weight': mod_weight.transpose((1, 0)),
        'conv.modulation.bias': mod_bias + 1,
        'noise.weight': np.array([noise]),
        'activate.bias': bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + '.conv.weight'] = torch.flip(
            dic_torch[target_name + '.conv.weight'], [3, 4]
        )

    return dic_torch


def convert_conv(vars, source_name, target_name, bias=True, start=0):
    weight = vars[source_name + '/weight'].value().eval()

    dic = {'weight': weight.transpose((3, 2, 0, 1))}

    if bias:
        dic['bias'] = vars[source_name + '/bias'].value().eval()

    dic_torch = {}

    dic_torch[target_name + f'.{start}.weight'] = torch.from_numpy(dic['weight'])

    if bias:
        dic_torch[target_name + f'.{start + 1}.bias'] = torch.from_numpy(dic['bias'])

    return dic_torch


def convert_torgb(vars, source_name, target_name):
    weight = vars[source_name + '/weight'].value().eval()
    mod_weight = vars[source_name + '/mod_weight'].value().eval()
    mod_bias = vars[source_name + '/mod_bias'].value().eval()
    bias = vars[source_name + '/bias'].value().eval()

    dic = {
        'conv.weight': np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        'conv.modulation.weight': mod_weight.transpose((1, 0)),
        'conv.modulation.bias': mod_bias + 1,
        'bias': bias.reshape((1, 3, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    return dic_torch


def convert_dense(vars, source_name, target_name):
    weight = vars[source_name + '/weight'].value().eval()
    bias = vars[source_name + '/bias'].value().eval()

    dic = {'weight': weight.transpose((1, 0)), 'bias': bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    return dic_torch


def update(state_dict, new):
    for k, v in new.items():
        if k not in state_dict:
            raise KeyError(k + ' is not found')

        if v.shape != state_dict[k].shape:
            raise ValueError(f'Shape mismatch: {v.shape} vs {state_dict[k].shape}')

        state_dict[k] = v


def discriminator_fill_statedict(statedict, vars, size):
    log_size = int(math.log(size, 2))

    update(statedict, convert_conv(vars, f'{size}x{size}/FromRGB', 'convs.0'))

    conv_i = 1

    for i in range(log_size - 2, 0, -1):
        reso = 4 * 2 ** i
        update(
            statedict,
            convert_conv(vars, f'{reso}x{reso}/Conv0', f'convs.{conv_i}.conv1'),
        )
        update(
            statedict,
            convert_conv(
                vars, f'{reso}x{reso}/Conv1_down', f'convs.{conv_i}.conv2', start=1
            ),
        )
        update(
            statedict,
            convert_conv(
                vars, f'{reso}x{reso}/Skip', f'convs.{conv_i}.skip', start=1, bias=False
            ),
        )
        conv_i += 1

    update(statedict, convert_conv(vars, f'4x4/Conv', 'final_conv'))
    update(statedict, convert_dense(vars, f'4x4/Dense0', 'final_linear.0'))
    update(statedict, convert_dense(vars, f'Output', 'final_linear.1'))

    return statedict


def fill_statedict(state_dict, vars, size, n_dense=8):
    log_size = int(math.log(size, 2))

    for i in range(n_dense):
        update(state_dict, convert_dense(vars, f'G_mapping/Dense{i}', f'style.{i + 1}'))

    update(
        state_dict,
        {
            'input.input': torch.from_numpy(
                vars['G_synthesis/4x4/Const/const'].value().eval()
            )
        },
    )

    update(state_dict, convert_torgb(vars, 'G_synthesis/4x4/ToRGB', 'to_rgb1'))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(vars, f'G_synthesis/{reso}x{reso}/ToRGB', f'to_rgbs.{i}'),
        )

    update(state_dict, convert_modconv(vars, 'G_synthesis/4x4/Conv', 'conv1'))

    conv_i = 0

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                vars,
                f'G_synthesis/{reso}x{reso}/Conv0_up',
                f'convs.{conv_i}',
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars, f'G_synthesis/{reso}x{reso}/Conv1', f'convs.{conv_i + 1}'
            ),
        )
        conv_i += 2

    for i in range(0, (log_size - 2) * 2 + 1):
        update(
            state_dict,
            {
                f'noises.noise_{i}': torch.from_numpy(
                    vars[f'G_synthesis/noise{i}'].value().eval()
                )
            },
        )

    update(
        state_dict,
        {
            'truncation_latent': torch.from_numpy(
                vars['dlatent_avg'].value().eval()
            )
        },
    )

    return state_dict


def get_dense_num(g_ema):
    n_dense = 0
    for key in g_ema.vars.keys():
        if 'G_mapping/Dense' in key:
            idx = key.find('Dense')
            n_dense = max(n_dense, int(key[idx + 5]))
    return n_dense + 1



if __name__ == '__main__':
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, default='./stylegan2-ada')
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--outdir', type=str, default='./output')
    parser.add_argument('path', metavar='PKL')

    args = parser.parse_args()

    sys.path.append(args.repo)

    import dnnlib
    from dnnlib import tflib

    tflib.init_tf()

    with open(args.path, 'rb') as fp:
        generator, discriminator, g_ema = pickle.load(fp)
    
    size = g_ema.output_shape[2]
    n_dense = get_dense_num(g_ema)
    g = Generator(size, args.latent_dim, n_dense, channel_multiplier=args.channel_multiplier)
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, g_ema.vars, size, n_dense)
    g.load_state_dict(state_dict)
    g = g.to(device)

    path = join(args.outdir, splitext(basename(args.path))[0] + '.onnx')
    latents = torch.rand(1, g.num_layers + 1, 512)
    truncation = torch.tensor(0.5)

    torch.onnx.export(
        model=g, 
        args=(latents, truncation),
        export_params=True,
        f=path,
        verbose=False,
        training=False,
        do_constant_folding=False,
        input_names=['latents', 'truncation'],
        output_names=['images'],
        opset_version=10
    )
