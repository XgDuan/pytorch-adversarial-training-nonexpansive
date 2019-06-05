import torch
import torch.nn as nn
import torch.nn.functional as F

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


class CRelu(nn.Module):
    def __init__(self):
        super(CRelu, self).__init__()

    def forward(self, x):
        """
        https://arxiv.org/pdf/1603.05201.pdf
        """
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)  # relu along channel


class NormPooling2d(nn.Module):
    def __init__(self, *k, **kw):
        super(NormPooling2d, self).__init__()
        self.avg_pool_kernel = nn.AvgPool2d(*k, **kw)

    def forward(self, x):
        ks = self.avg_pool_kernel.kernel_size
        coef = __import__('math').sqrt(ks[0] * ks[1])
        return coef * torch.sqrt(self.avg_pool_kernel(x * x))


class Model(nn.Module):
    def __init__(self, i_c=1, n_c=10, v=1, z=1, use_bias=False, v_use_grad=False):
        super(Model, self).__init__()
        self.relu_expa = 2

        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=use_bias)
        self.crelu1 = CRelu()
        self.pool1 = NormPooling2d((2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(32 * self.relu_expa, 64, 5, stride=1, padding=2, bias=use_bias)
        self.crelu2 = CRelu()
        self.pool2 = NormPooling2d((2, 2), stride=(2, 2), padding=0)

        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(7 * 7 * 64 * self.relu_expa, 1024, bias=use_bias)
        self.crelu3 = CRelu()
        self.fc2 = nn.Linear(1024 * self.relu_expa, n_c, bias=use_bias)
        self.u = nn.Parameter(torch.rand(n_c), requires_grad=True)
        self.v = nn.Parameter(torch.tensor(v), requires_grad=v_use_grad)
        self.z = torch.tensor(z, requires_grad=False)

    def forward(self, x_i, _eval=False):

        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        x_o = self.conv1(x_i / 32)
        x_o = self.crelu1(x_o)
        x_o = self.pool1(x_o)

        x_o = self.conv2(x_o / 64)
        x_o = self.crelu2(x_o)
        x_o = self.pool2(x_o)

        x_o = self.flatten(x_o)

        x_o = self.crelu3(self.fc1(x_o))

        self.train()

        return self.fc2(x_o)

    def loss_a(self, pred, label):
        return F.cross_entropy(pred * self.u, label)

    def loss_b(self, pred, label):
        return F.cross_entropy(pred * self.v, label)

    def loss_c(self, pred, label):
        digit = F.softmax(pred * self.z)
        digit = torch.gather(digit, 1, label.view(-1, 1))
        return torch.mean(torch.log(1 - digit + 1e-10)) / self.z

    def loss_nonexpa(self):
        loss_list = []

        def l_loss(m):
            sum_over_axis1 = torch.sum(torch.abs(m), 1) - 1
            sum_over_axis1 = torch.cat([sum_over_axis1, torch.zeros_like(sum_over_axis1)], dim=-1)
            return torch.sum(torch.max(sum_over_axis1))

        for name, val in self.state_dict().items():
            if name.endswith('weight'):
                val_mat = val.reshape(val.shape[0], -1)
                val_mat_t = torch.transpose(val_mat, 0, 1)
                # import ipdb; ipdb.set_trace()
                loss_list.append(torch.min(l_loss(torch.matmul(val_mat, val_mat_t)), l_loss(torch.matmul(val_mat_t, val_mat))))

        return torch.sum(torch.stack(loss_list))

if __name__ == '__main__':
    i = torch.FloatTensor(4, 1, 28, 28)

    n = Model()
    print(n(i).size())

    l = n.loss_nonexpa()
