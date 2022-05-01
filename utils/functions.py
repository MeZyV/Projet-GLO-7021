import torch

def topk_2d(x, k):
    # Return the top k elements of 2d tensor x along all dimensions.
    indices = torch.tensor([
        [i, j] for i in range(x.shape[0]) for j in range(x.shape[1])
    ])
    topk_val, topk_ind = torch.topk(x.flatten(), k, dim=0)
    return topk_val, indices[topk_ind]

if __name__ == '__main__':
    X = torch.tensor([
        [2, 0, 1, 0, 0],
        [0, 2, 0, 1, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [10, 0, 0, 0, 1],
    ])

    print(topk_2d(X, 5))