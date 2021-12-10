import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
f = h5py.File('test_outputs/scalar/scalar_s1/scalar_s1_p0.h5')
ke = np.array(f['tasks/KE'])[:,0,0,0]
time = np.array(f['scales/sim_time'])

i = np.argmin(np.abs(time-100))
i2 = np.argmin(np.abs(time-300))
#p = np.polyfit(time[i:i2],np.log(ke[i:i2]),1)
#poly = np.poly1d(p)
print(ke)
plt.clf()

plt.semilogy(time,ke)
#plt.plot(time,np.exp(poly(time)),label="gamma=%s"%float('%.1g' % p[0]),linestyle='--')
plt.xlabel("time")
#plt.xlim(0,200)
#plt.ylim(1e-5,1e-2)
plt.ylabel('KE')
plt.legend()
plt.savefig('ke.pdf')
