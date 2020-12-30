import numpy as np
import torch
import torch.fft as fft
import complex_numbers as cn
import torch.autograd.profiler as profiler
import util


def compare_complex(old, new):
    # if old.shape == new.shape:
    #     print('Shapes equal')
    # else:
    #     print('Shapes are different:')
    #     print(old.shape)
    #     print(new.shape)
    #
    # if old.dtype == new.dtype:
    #     print('Dtypes are equal')
    # else:
    #     print('Dtypes are different:')
    #     print(old.dtype)
    #     print(new.dtype)

    print(f"View_as_complex: {new.allclose(torch.view_as_complex(old))}")
    print(f"View_as_real: {torch.view_as_real(new).allclose(old)}, diff={torch.sum(torch.abs(torch.view_as_real(new) - old))}")

    # rold = old[..., 0]
    # iold = old[..., 1]
    # rnew = torch.real(new)
    # inew = torch.imag(new)
    #
    # rdiff = torch.sum(torch.abs(rold - rnew))
    # idiff = torch.sum(torch.abs(iold - inew))
    #
    # real_equal = rold.allclose(rnew)
    # imag_equal = iold.allclose(inew)
    # print(f"Real/Imag: {real_equal}/{imag_equal}, Diff: {rdiff},{idiff}")


def compare(old, new, drop_last):
    if drop_last:
        old = old.squeeze(4)

    if old.shape == new.shape:
        print('Shapes equal')
    else:
        print('Shapes are different:')
        print(old.shape)
        print(new.shape)

    if old.dtype == new.dtype:
        print('Dtypes are equal')
    else:
        print('Dtypes are different:')
        print(old.dtype)
        print(new.dtype)

    print(torch.all(old.eq(new)))


def fft_comparison():
    global a, b
    t = torch.rand((42, 32, 121, 121))
    a = torch.rfft(t, signal_ndim=2)
    b = fft.rfftn(t, dim=[-2, -1])
    compare_complex(a, b)


def fft_new(z, x, label):
    zf = fft.rfftn(z, dim=[-2, -1])
    xf = fft.rfftn(x, dim=[-2, -1])

    # R[batch, 1, 121, 61]
    kzzf = torch.sum(torch.real(zf) ** 2 + torch.imag(zf) ** 2, dim=1, keepdim=True)

    # C[batch, 1, 121, 61]
    t = xf * torch.conj(zf)
    kxzf = torch.sum(t, dim=1, keepdim=True)

    # C[batch, 1, 121, 61, 2]
    alphaf = label.to(device=z.device) / (kzzf + lambda0)

    # R[batch, 1, 121, 121]
    return fft.irfftn(kxzf * alphaf, s=[121, 121], dim=[-2, -1])


def fft_old(z, x, label):
    # [batch, 32, 121, 61, 2]
    zf = torch.rfft(z, signal_ndim=2)
    xf = torch.rfft(x, signal_ndim=2)

    # [batch, 1, 121, 61, 1]
    kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)

    # [batch, 1, 121, 61, 2]
    t = cn.mulconj(xf, zf)
    kxzf = torch.sum(t, dim=1, keepdim=True)

    # [batch, 1, 121, 61, 2]
    alphaf = label.to(device=z.device) / (kzzf + lambda0)

    # [batch, 1, 121, 121]
    return torch.irfft(cn.mul(kxzf, alphaf), signal_ndim=2)


##############################################


# fft_comparison()

##############################################

lambda0 = 1e-4
y = util.gaussian_shaped_labels(4.166666666666667, [121, 121]).astype(np.float32)

x = torch.rand((42, 32, 121, 121))
z = torch.rand((42, 32, 121, 121))
fft_label_view = torch.Tensor(y).view(1, 1, 121, 121).cuda()
label_old = torch.rfft(fft_label_view, signal_ndim=2).repeat(42, 1, 1, 1, 1).cuda(non_blocking=True)
label_new = fft.rfftn(fft_label_view, dim=[-2, -1]).repeat(42, 1, 1, 1).cuda(non_blocking=True)

##############################################
zfnew = fft.rfftn(z, dim=[-2, -1])
zfold = torch.rfft(z, signal_ndim=2)

xfnew = fft.rfftn(x, dim=[-2, -1])
xfold = torch.rfft(x, signal_ndim=2)

tnew = xfnew * torch.conj(zfnew)
# told = cn.mulconj(xfold, zfold)
# rtold = told[...,0]
# itold = told[...,1]
# tnew = torch.complex(rtold, itold)
# told = torch.view_as_complex(tnew)
told1 = torch.view_as_real(tnew)
told2 = torch.stack([torch.real(tnew), torch.imag(tnew)], dim=4)
print('t difference: ')
compare_complex(told1, tnew)
compare_complex(told2, tnew)


print('kxzf difference: ')
compare_complex(torch.sum(told1, dim=1, keepdim=True), torch.sum(tnew, dim=1, keepdim=True))
compare_complex(torch.sum(told2, dim=1, keepdim=True), torch.sum(tnew, dim=1, keepdim=True))

exit()
##############################################


with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        a = fft_old(z, x, label_old)
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        b = fft_new(z, x, label_new)
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
if b.dtype == torch.complex64:
    compare_complex(a, b)
else:
    compare(a, b, True)

# print(f'Result shapes: old: {a.shape}, new: {b.shape}')
# print(f'Result dtype: old: {a.dtype}, new: {b.dtype}')
# print("Results equal: {}".format(torch.all(a.eq(b))))
