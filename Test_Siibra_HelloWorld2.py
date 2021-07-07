# =================================================================
# Examnples of siibra in action, to test it works.
#
# Taken from
# https://github.com/FZJ-INM1-BDA/siibra-python/blob/main/examples/Walkthrough.ipynb
# =================================================================
import siibra
from nilearn import plotting,image
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

baseInPath = 'Data_Raw/siibra'

def connectSiibra():
    import webbrowser
    webbrowser.open('https://nexus-iam.humanbrainproject.org/v0/oauth2/authorize')
    token = input("Enter your token here, then press 'Enter': ")
    siibra.ebrains.set_token(token)

def handleCache():
    from os import environ
    # -- #!mkdir -p /tmp/siibracache
    environ['SIIBRA_CACHEDIR'] = "Data_Raw/siibra"
    # siibra.clear_cache()

# This is a test code to check whether we can access a specific (open) dataset
def main():
    handleCache()
    connectSiibra()
    # Accessing parcellation maps
    atlas = siibra.atlases["human"]
    # atlas.select(parcellation="julich-brain 2.5")
    # in MNI 152 space
    icbm_map = atlas.get_map(space="mni152")
    # the julich brain map comes in separate l/r hemispheres,
    # so we iterate over all maps.
    for m in icbm_map.fetchall():
        plotting.plot_stat_map(m)

    # bigbrain
    reso_mm = 0.64
    bigbrain_tpl = atlas.get_template("bigbrain")
    bigbrain_map = atlas.get_map(space="bigbrain")
    plotting.plot_stat_map(bigbrain_map.fetch(reso_mm),
                           bigbrain_tpl.fetch(reso_mm) )

    # DK atlas
    atlas.select(parcellation="desikan")
    dk_map = atlas.get_map(space="mni152")
    plotting.plot_stat_map(dk_map.fetch(), cmap=plt.cm.tab10)
    plotting.show()


if __name__ == '__main__':
    # main()
    main()
