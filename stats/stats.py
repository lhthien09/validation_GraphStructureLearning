import os
import torch
from argparse import ArgumentParser

def compute_statistics(params):
    dataset = params.dataset.lower()
    # assume calling script in the root project
    save_path = os.path.join(os.getcwd(), "stats")

    dgm_layers = params.dgm_layers
    # check if it's empty, for case #dgmlayers=0
    if dgm_layers[0]:
        emb_dims=dgm_layers[0][2]
    else:
        emb_dims=0

    # get #dgm_layers, exclude empty elements
    dgm_num = sum(1 for x in dgm_layers if x)

    paths = {
        "dataset": params.dataset,
        "k": params.k,
        "dist": params.distance,
        "dims": emb_dims,
        "dgm": dgm_num,
        "f": params.ffun,
        "g": params.gfun
    }

    file_path = os.path.join(save_path,f'{paths["dataset"]}_test_acc_'\
                        f'k_{paths["k"]}_{paths["dist"]}_'\
                        f'd_{paths["dims"]}_'\
                        f'dgm_{paths["dgm"]}_' \
                        f'f_{paths["f"]}_' \
                        f'g_{paths["g"]}.txt'.lower())

    with open(file_path, "r") as f:
        test_acc = [float(line.strip()) for line in f.readlines()]
    tensor = torch.tensor(test_acc)

    with open(file_path, "a+") as f:
        f.write(f"mean: {str(tensor.mean().item())}\n")
        f.write(f"std: {str(tensor.std().item())}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--k", default=5)
    parser.add_argument("--distance", default="euclidean")
    parser.add_argument("--dim", default=4)
    parser.add_argument("--dgm_layers", default= [[32,16,4],[],[]], type=lambda x :eval(x))
    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')

    params = parser.parse_args()

    compute_statistics(params)