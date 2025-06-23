import mdtraj as md
import matplotlib.pyplot as plt

traj = md.load('traj.xtc', top='topol.pdb')

densities = md.density(traj)

plt.plot(traj.time / 1000, densities)  
plt.xlabel('Time (ns)')
plt.ylabel('Density (g/mL)')
plt.title('Average Mass Density per Frame')
plt.show()