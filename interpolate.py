import scipy


x = scipy.io.loadmat('output/dof_curve_interpolated.mat')
print(x['x'])