from mountaincar import *

mc = MountainCar()
mc.apply_force(+1)
mc.simulate_timesteps(n=100, dt=0.01)
print (mc.x, mc.x_d, mc.R)

mv = MountainCarViewer(mc)
mv.create_figure(n_steps = 200, max_time = 200)
plb.show()
for n in range(200):
    mc.simulate_timesteps(100, 0.01)
    mv.update_figure()
    plb.show()


mc.reset()