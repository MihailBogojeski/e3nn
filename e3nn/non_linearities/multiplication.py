# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import SO3


class Multiplication(torch.nn.Module):
    """
    Multiplication

    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(2, 1), (5, 0), (1, 1), (1, 2), (2, 1)]
    """
    def __init__(self, Rs_1, Rs_2, get_l_output=None):
        super().__init__()

        def get_ls(l_1, l_2):
            return list(range(abs(l_1 - l_2), l_1 + l_2 + 1))

        self.get_ls = get_l_output if get_l_output is not None else get_ls

        self.Rs_1 = SO3.normalizeRs(Rs_1)
        self.Rs_2 = SO3.normalizeRs(Rs_2)

        Rs_out = []
        for mul_1, l_1, p_1 in self.Rs_1:
            for mul_2, l_2, p_2 in self.Rs_2:
                for l in self.get_ls(l_1, l_2):
                    Rs_out.append((mul_1 * mul_2, l, p_1 * p_2))

                    if l_1 == 0 or l_2 == 0:
                        C = SO3.clebsch_gordan(l, l_1, l_2, dtype=torch.float64).view(2 * l + 1, 2 * l + 1) * (2 * l + 1) ** 0.5
                        assert (C - torch.eye(2 * l + 1, dtype=torch.float64)).abs().max() < 1e-10, C.numpy().round(3)

        self.Rs_out = SO3.normalizeRs(Rs_out)


    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.view(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.view(-1, n_2)
        assert size_1 == size_2
        batch = features_1.size(0)

        output = []
        index_1 = 0
        for mul_1, l_1, _p_1 in self.Rs_1:
            f_1 = features_1.narrow(1, index_1, mul_1 * (2 * l_1 + 1)).reshape(batch, mul_1, 2 * l_1 + 1)  # [z, u, j]
            index_1 += mul_1 * (2 * l_1 + 1)

            index_2 = 0
            for mul_2, l_2, _p_2 in self.Rs_2:

                f_2 = features_2.narrow(1, index_2, mul_2 * (2 * l_2 + 1)).reshape(batch, mul_2, 2 * l_2 + 1)  # [z, v, k]
                index_2 += mul_2 * (2 * l_2 + 1)

                for l in self.get_ls(l_1, l_2):
                    C = SO3.clebsch_gordan(l, l_1, l_2, cached=True, like=f_1) * (2 * l + 1) ** 0.5
                    out = torch.einsum("ijk,zuj,zvk->zuvi", C, f_1, f_2).view(batch, mul_1 * mul_2 * (2 * l + 1))  # [z, w, i]
                    output.append(out)
        assert index_1 == features_1.size(1)
        assert index_2 == features_2.size(1)

        output = torch.cat(output, dim=1)
        return output.view(*size_1, -1)


class ElementwiseMultiplication(torch.nn.Module):
    """
    Elementwise multiplication

    [(2, 0), (1, 1)] x [(1, 1), (2, 0)] = [(1, 1), (1, 0), (1, 1)]
    """
    def __init__(self, Rs_1, Rs_2):
        super().__init__()

        Rs_1 = SO3.normalizeRs(Rs_1)
        Rs_2 = SO3.normalizeRs(Rs_2)
        assert sum(mul for mul, _, _ in Rs_1) == sum(mul for mul, _, _ in Rs_2)

        i = 0
        while i < len(Rs_1):
            mul_1, l_1, p_1 = Rs_1[i]
            mul_2, l_2, p_2 = Rs_2[i]

            if mul_1 < mul_2:
                Rs_2[i] = (mul_1, l_2, p_2)
                Rs_2.insert(i + 1, (mul_2 - mul_1, l_2, p_2))

            if mul_2 < mul_1:
                Rs_1[i] = (mul_2, l_1, p_1)
                Rs_1.insert(i + 1, (mul_1 - mul_2, l_1, p_1))
            i += 1

        self.Rs_1 = Rs_1
        self.Rs_2 = Rs_2

        Rs_out = []
        for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_1, Rs_2):
            assert mul == mul_2
            for l in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                Rs_out.append((mul, l, p_1 * p_2))

                if l_1 == 0 or l_2 == 0:
                    C = SO3.clebsch_gordan(l, l_1, l_2, dtype=torch.float64).view(2 * l + 1, 2 * l + 1) * (2 * l + 1) ** 0.5
                    assert (C - torch.eye(2 * l + 1, dtype=torch.float64)).abs().max() < 1e-10, C.numpy().round(3)

        self.Rs_out = Rs_out


    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.view(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.view(-1, n_2)
        assert size_1 == size_2
        batch = features_1.size(0)

        output = []
        index_1 = 0
        index_2 = 0
        for (mul, l_1, _), (_, l_2, _) in zip(self.Rs_1, self.Rs_2):
            f_1 = features_1.narrow(1, index_1, mul * (2 * l_1 + 1)).reshape(batch * mul, 2 * l_1 + 1)  # [z, j]
            index_1 += mul * (2 * l_1 + 1)

            f_2 = features_2.narrow(1, index_2, mul * (2 * l_2 + 1)).reshape(batch * mul, 2 * l_2 + 1)  # [z, k]
            index_2 += mul * (2 * l_2 + 1)

            if l_1 == 0 or l_2 == 0:
                # special case optimization
                out = f_1 * f_2
                output.append(out.view(batch, mul * (2 * max(l_1, l_2) + 1)))
            else:
                for l in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                    C = SO3.clebsch_gordan(l, l_1, l_2, cached=True, like=f_1) * (2 * l + 1) ** 0.5
                    out = torch.einsum("ijk,zj,zk->zi", C, f_1, f_2).view(batch, mul * (2 * l + 1))  # [z, i]
                    output.append(out)
        assert index_1 == features_1.size(1)
        assert index_2 == features_2.size(1)

        output = torch.cat(output, dim=1)
        return output.view(*size_1, -1)
