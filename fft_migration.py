import torch
import torch.fft as fft




t = torch.rand((42,32,121,121))

a = torch.rfft(t, signal_ndim=2)

b = fft.rfftn(t, dim=[-2,-1])

print("Real parts equal: {}".format(torch.all(a[...,0].eq(torch.real(b)))))
print("Imag parts equal: {}".format(torch.all(a[...,1].eq(torch.imag(b)))))
