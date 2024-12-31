# --------------------------------------------------------------------------------------
# WholeBrain base folder!
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
from sys import platform

if platform == "win32":
    WorkBrainDataFolder = "L:/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WorkBrain/_Data_Raw/"
elif platform == "darwin":
    WorkBrainDataFolder = "/Users/dagush/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WorkBrain/_Data_Raw/"
else:
    raise Exception('Unrecognized OS!!!')
