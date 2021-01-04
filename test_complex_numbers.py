import torch
import torch.fft as fft


# a = torch.rand((1000, 1000), dtype=torch.complex64)
# # b = torch.view_as_real(a)
# b = torch.stack([torch.real(a), torch.imag(a)], dim=2)
#
# print(f"View_as_complex: {a.allclose(torch.view_as_complex(b))}")
# print(f"View_as_real: {torch.view_as_real(a).allclose(b)}")
# print(f"View_as_complex: {torch.sum(a, dim=1,keepdim=True).allclose(torch.view_as_complex(torch.sum(b, dim=1,keepdim=True)))}")
# print(f"View_as_real: {torch.view_as_real(torch.sum(a, dim=1,keepdim=True)).allclose(torch.sum(b, dim=1,keepdim=True))}")
#

def compare_complex(old, new):
    diff = torch.sum(torch.abs(torch.view_as_real(new) - old))
    print(f"View_as_complex: {new.allclose(torch.view_as_complex(old))}, "
          f"View_as_real: {torch.view_as_real(new).allclose(old)}, "
          f"Diff={diff}")


# # shape = (42,32,121,121)
# shape = (42, 32, 10, 10)
# x = torch.rand(shape)
# z = torch.rand(shape)
#
# zfnew = fft.rfftn(z, dim=[-2, -1])
# xfnew = fft.rfftn(x, dim=[-2, -1])
#
# # zfold = torch.rfft(z, signal_ndim=2)
# # xfold = torch.rfft(x, signal_ndim=2)
#
# tnew = xfnew * torch.conj(zfnew)
#
# told1 = torch.view_as_real(tnew)
# told2 = torch.stack([torch.real(tnew), torch.imag(tnew)], dim=4)
# told3 = torch.stack([tnew.real, tnew.imag], dim=4)
#
# print(told1.dtype)
# print(told2.dtype)
# print(told3.dtype)
#
# print(torch.all(told1 == told2))
#
# print('t difference: ')
# compare_complex(told1, tnew)
# compare_complex(told2, tnew)
# compare_complex(told3, tnew)
#
# print('kxzf difference: ')
# compare_complex(torch.sum(told1, dim=1, keepdim=True), torch.sum(tnew, dim=1, keepdim=True))
# compare_complex(torch.sum(told2, dim=1, keepdim=True), torch.sum(tnew, dim=1, keepdim=True))
# compare_complex(torch.sum(told3, dim=1, keepdim=True), torch.sum(tnew, dim=1, keepdim=True))

shape = (42, 32, 10, 10)
x = torch.randn(shape)
z = torch.randn(shape)

tnew = fft.rfftn(x, dim=[2, 3]) * torch.conj(fft.rfftn(z, dim=[2, 3]))
# tnew2 = torch.randn(shape, dtype=torch.complex64)
# tnew2 = torch.empty(shape, dtype=torch.complex64).normal_(mean=0, std=0.00001)

# h1 = torch.histc(torch.view_as_real(tnew))
# h2 = torch.histc(torch.view_as_real(tnew2))
#
# import matplotlib.pyplot as plt
# plt.plot(h1)
# plt.plot(h2)
# plt.show()

sum_float = torch.sum(torch.view_as_real(tnew), dim=1, keepdim=True)
sum_complex = torch.sum(tnew, dim=1, keepdim=True)

print('Equality as real:    ', torch.count_nonzero(torch.eq(sum_float, torch.view_as_real(sum_complex))).item())
print('Equality as complex: ', torch.count_nonzero(torch.eq(torch.view_as_complex(sum_float), sum_complex)).item())
print('Compared as real:    ', torch.allclose(sum_float, torch.view_as_real(sum_complex)))
print('Compared as complex: ', torch.allclose(torch.view_as_complex(sum_float), sum_complex))