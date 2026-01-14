from src.mutiresnetunet import MultiResUnet


def count_parameters(model):
    """
    计算给定模型的总参数量
    :param model: PyTorch模型（torch.nn.Module）
    :return: 模型的总参数量（int）
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 返回总参数量
    return total_params


if __name__ == '__main__':

    model = MultiResUnet(in_channels=1, out_channels=1)

    total_params = count_parameters(model)
    print(f'Total number of parameters: {total_params:,}')
