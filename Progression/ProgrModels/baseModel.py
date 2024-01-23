# --------------------------------------------------------------------------------------
# Base class for progression models...
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
import warnings


class base_progression:
    def __init__(self):
        pass

    def serParms(self, parms):
        for parm in parms:
            if hasattr(self, parm):
                setattr(self, parm, parms[parm])  # no problem with shadowing, we do not have state variables here!
            else:
                warnings.warn(f'parameter undefined: {parm} (perhaps not needed?)')

    def getParm(self, parm):
        return getattr(self, parm)

    # def recompileSignatures(self):
    #     self.dfun.recompile()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF