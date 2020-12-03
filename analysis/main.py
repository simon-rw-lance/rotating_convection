from rot_toolkit import SimData

sim = "r14500"
folder = "../IVP/EVP-test/n1_t1e5/pert_s/"
sim_id = sim

direc = f"{folder}{sim}/raw_data/"
save  = f"{folder}/{sim}/figs/"
save  = f"figs/{sim}/"
data = SimData(direc, save, name=sim, id=sim_id, making_figs=True)

print(data.test)
