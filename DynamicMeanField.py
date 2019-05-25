
# Implementation of the Dynamic Mean Field model, also known as FIC (Feedback Inhibition control)
#
# Presented at
# Deco et al. (2014) J Neurosci.
# http://www.jneurosci.org/content/34/23/7886.long


# transfer functions:
# transfer function: excitatory
#--------------------------------------------------------------------------
ae=310;
be=125;
de=0.16;
def He(x):
    y = (ae*x-be)  # Use y=(ae*x-be)*(1+Receptor*wgaine) for LSD
    if (y != 0):
        return y/(1-np.exp(-de*y))
    else:
        return 0

# transfer function: inhibitory
#--------------------------------------------------------------------------
ai=615;
bi=177;
di=0.087;
def Hi(x):
    y = (ai*x-bi)  # Use y=(ai*x-bi)*(1+Receptor*wgaini) for LSD
    if (y != 0):
        return y/(1-np.exp(-di*y))
    else:
        return 0
