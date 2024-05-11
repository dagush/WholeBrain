import numpy as np
import nibabel as nib

def printMinMax(name, vtx):
    print(name)
    for id in range(3):
        print(f'   Min {id}:{np.min(vtx[:, id])}  |  Max {id}:{np.max(vtx[:, id])}')


def testGlasser():
    Glasser360_baseFolder = "../../Data_Raw/Parcellations/Glasser360/"
    # =============== Load the geometry ==================
    glassers_L = nib.load(Glasser360_baseFolder + 'Glasser360.L.mid.32k_fs_LR.surf.gii')
    vtx_L, tri_L = glassers_L.agg_data()
    glassers_R = nib.load(Glasser360_baseFolder + 'Glasser360.R.mid.32k_fs_LR.surf.gii')
    vtx_R, tri_R = glassers_R.agg_data()

    printMinMax('LEFT', vtx_L)
    printMinMax('RIGHT', vtx_R)


dbs80_baseFolder = "../../Data_Raw/Parcellations/dbs80/"
dbs80_L = nib.load(dbs80_baseFolder + 'dbs80_left.func.gii')
vtx_L, tri_L = dbs80_L.agg_data()

print('done')