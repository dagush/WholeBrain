# --------------------------------------------------------------------------------------
# WholeBrain base folder!
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
from sys import platform

if platform == "win32":
    WholeBrainFolder = "L:/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WholeBrain/"
elif platform == "darwin":
    WholeBrainFolder = "/Users/dagush/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WholeBrain/"
else:
    raise Exception('Unrecognized OS!!!')
