# --------------------------------------------------------------------------------------
# Base class for progression models...
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------


class base_progression:
    def __init__(self):
        pass

    def serParms(self, parms):
        for parm in parms:
            setattr(self, parm, parms[parm])  # no problem with shadowing, we do not have state variables here!

    def getParm(self, parm):
        return getattr(self, parm)

    # def recompileSignatures(self):
    #     self.dfun.recompile()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF