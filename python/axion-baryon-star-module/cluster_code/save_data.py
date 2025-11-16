import numpy as np



def save_total_data_binary(Fdata, Gdata, nNdata, Udata, Adata, Pdata, times, radii, NSmasses, NSradii, preamblevars, directory):
    # preamblevars will contain the following variables
    # [Rcentral in m^-2, fa in GeV, ma in GeV, epsilon, INTERMEDIATEMETRICINTEGRATOR]
    myheader = 'Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3])

    np.save(directory + "/F-solution" + ".npy", Fdata)
    print("saved F")

    np.save(directory + "/G-solution" + ".npy", Gdata)
    print("saved G")

    np.save(directory + "/nN-solution" + ".npy", nNdata)
    print("saved nN")

    np.save(directory + "/U-solution" + ".npy", Udata)
    print("saved U")

    np.save(directory + "/A-solution" + ".npy", Adata)
    print("saved A")

    np.save(directory + "/P-solution" + ".npy", Pdata)
    print("saved P")

    np.save(directory + "/times" + ".npy", times)
    np.save(directory + "/radii" + ".npy", radii)
    np.save(directory + "/NSmasses" + ".npy", NSmasses)
    np.save(directory + "/NSradii" + ".npy", NSradii)
    print("saved times and radii")

    np.savetxt(directory + "/info.txt", np.ones(10), header=myheader)
    print("saved header")


    return;
