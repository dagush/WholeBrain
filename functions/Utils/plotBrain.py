# =================================================================
#  Plotting brain functions with
#    NiBabel for the gifti files, and
#    matplotlib for plotting
# =================================================================
# =================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh, PolyCollection
# Import lighting object for shading surface plots.
from matplotlib.colors import LightSource


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr


def computeFaceNormals(vertices, faces):
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # Change - normals are per vertex, so I made it per face.
    # normalsarray = np.array([np.array((np.sum(norm[face[:], 0]/3), np.sum(norm[face[:], 1]/3), np.sum(norm[face[:], 2]/3))/np.sqrt(np.sum(norm[face[:], 0]/3)**2 + np.sum(norm[face[:], 1]/3)**2 + np.sum(norm[face[:], 2]/3)**2)) for face in faces])
    return n


def computeVertexNormals(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)

    n = computeFaceNormals(vertices, faces)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    return norm


# =================================================================
#  Map values to colors...
# =================================================================
def mapValues2Colors(values, cmap):
    # Convert values to colors: normalize and map to a color palette
    norm = plt.Normalize(values.min(), values.max())
    colors = cmap(norm(values))
    return colors


# =================================================================
# Functions to compute values per vertex and per face
# =================================================================
def computeFaceValues(triangles, vertexValues):
    # Change - colors are per vertex, so I make them per face.
    colorsarray = np.array([np.sum(vertexValues[face[:]]/3) for face in triangles])
    return colorsarray


def computeVertexValues(vertices, parcellationMap, parcellationValues):
    surfaceValues = np.zeros(len(vertices))
    # surfaceValuesR = np.zeros_like()

    # left hemisphere
    for i in range(0,360):
        idx = np.where(parcellationMap == i)
        surfaceValues[idx] = parcellationValues[i]
    return surfaceValues


# =================================================================
# Plot a single view using Poly3DCollection
# =================================================================
# def plotView(ax,  # The axis where to plot to
#              surface3DModel,  # The 3D surface model to plot to. Usually, a 16k- or 32k-vertex surface, in a standard .gii format
#              surfaceValues,  # The values to plot on the surface. Usually, 16k or 32k values, one for each VERTEX, in a standard .gii format
#              cmap,  # The color map to use to paint the atlas values
#              c_azimuth, c_altitude,  # Camera azimuth and altitude
#              l_azimuth, l_altitude,   # Light azimuth and altitude
#              flatData = False
#              ):
#     # First, let's extract the vertices and triangles for the 3D model.
#     vertices, triangles = surface3DModel.agg_data()
#     # x,y,z = vertices.T
#     # Normalised...
#     if not flatData:
#         # scaledVert = vertices * np.array([1,1,1])
#         verticesLnorm = (vertices - np.min(vertices, axis=0))/np.ptp(vertices, axis=0)
#     else:
#         ptp = np.array([np.ptp(vertices[:,0:2], axis=0)[0], np.ptp(vertices[:,0:2], axis=0)[1], 1])
#         verticesLnorm = (vertices - np.min(vertices, axis=0)) / ptp
#     # verticesLnorm += np.array([1.0, 0, 0])
#     # verticesLnorm *= 1.5
#     print(f"min => {np.min(verticesLnorm, axis=0)}")
#     print(f"max => {np.max(verticesLnorm, axis=0)}")
#
#     # Set view parameters for subplot.
#     ax.view_init(c_altitude, c_azimuth)
#
#     colors = mapValues2Colors(computeFaceValues(triangles, surfaceValues), cmap)
#     normals = computeFaceNormals(verticesLnorm, triangles)
#
#     # Create a light source object for light from
#     # azimuth (from north), elevation (from 0 elevation plane). Both in degrees.
#     light = LightSource(l_azimuth, l_altitude)
#     lightingBias = 0.2
#     shaded = lightingBias + light.shade_normals(normals) * (1-lightingBias)
#     shadedColors = (colors[:,0:3].T * shaded).T
#
#     # pcL = art3d.Poly3DCollection(verticesLnorm[triangles], facecolors=colors)  # direct colors
#     pcL = art3d.Poly3DCollection(verticesLnorm[triangles], facecolors=shadedColors)  # shaded colors
#     ax.add_collection(pcL)
#     # ax.tripcolor(v, z, cmap=cmap)
#
#     ax.grid(False)
#     ax.axis('off')


# =================================================================
# Utility function to compute the surface values and then call plotView
# =================================================================
# def plotFuncView(ax,  # The axis where to plot to
#                  surface3DModel,  # The 3D surface model to plot to. Usually, a 16k- or 32k-vertex surface, in a standard .gii format
#                  parcellationValues,  # The values to plot on the surface, in parcellation space
#                  parcellationMap,  # The mapping from the parcellation to the model to plot
#                  cmap,  # The color map to use to paint the atlas values
#                  c_azimuth, c_altitude,  # Camera azimuth and altitude
#                  l_azimuth, l_altitude,   # Light azimuth and altitude
#                  flatData = False
#                  ):
#     surfaceValues = computeVertexValues(surface3DModel.agg_data('NIFTI_INTENT_POINTSET'), parcellationMap, parcellationValues)
#     plotView(ax, surface3DModel, surfaceValues, cmap, c_azimuth, c_altitude, l_azimuth, l_altitude, flatData=flatData)


# =================================================================
# =================================================================
# =================================================================

# =================================================================
# Functions to plot views: plotColorView is identical to the
# plotView above, but using tripcolor
#
# Adapted from TVB,
# https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/tutorials/utils.py
# =================================================================
def plotColorView(ax, cortex, data, viewkey,
                  shaded=False, shadowed=False, cmap=plt.cm.coolwarm,
                  l_azimuth=0, l_altitude=0, lightingBias=0.2,  # Light azimuth and altitude, and bias
                  zlim=None, zthresh=None, suptitle='', viewlabel=False):
    # ================= This part should be computed only once for all views, but this way it is easier...
    if 'flat' not in viewkey:
        vtx_L, tri_L = cortex['model_L'].agg_data()
        vtx_R, tri_R = cortex['model_R'].agg_data()
    else:
        vtx_L, tri_L = cortex['flat_L'].agg_data()
        vtx_R, tri_R = cortex['flat_R'].agg_data()
    xL, yL, zL = vtx_L.T
    xR, yR, zR = vtx_R.T

    if 'map_L' in cortex or 'map_R' in cortex:
        # if we are giving a mapping between regions and vertices, let's use it!
        rm_L = cortex['map_L']
        rm_R = cortex['map_R']
        if viewkey in ['Lh-lateral', 'Lh-medial', 'L-superior']:
            vvalues = computeVertexValues(vtx_L, rm_L, data['func_L'])
        else:
            vvalues = computeVertexValues(vtx_R, rm_R, data['func_R'])
    else:  # if not, it is because we were given the vertex-level values directly
        if viewkey in ['Lh-lateral', 'Lh-medial', 'L-superior']:
            vvalues = data['func_L']
        else:
            vvalues = data['func_R']

    views = {
        'Lh-lateral': Triangulation(-yL, zL, tri_L),  #tri[np.argsort(lh_ty)[::-1]]),
        'Lh-medial': Triangulation(yL, zL, tri_L[::-1]),  #lh_tri[np.argsort(lh_ty)]),
        'Rh-medial': Triangulation(-yR, zR, tri_R[::-1]),  #rh_tri[np.argsort(rh_ty)[::-1]]),
        'Rh-lateral': Triangulation(yR, zR, tri_R),  #rh_tri[np.argsort(rh_ty)]),
        'L-superior': Triangulation(xL, yL, tri_L),  #tri[np.argsort(tz)]),
        'R-superior': Triangulation(xR, yR, tri_R),  #tri[np.argsort(tz)]),
        'L-flat': Triangulation(xL, yL, tri_L),
        'R-flat': Triangulation(xR, yR, tri_R),
    }

    # ================= View-specific code...
    v = views[viewkey]
    if not viewlabel:
        plt.axis('off')
    if zthresh:
        z = vvalues.copy() * (abs(vvalues) > zthresh)
    # ================= Let's render it!
    if not shadowed or 'flat' in viewkey:
        if 'flat' in viewkey:
            kwargs = {'shading': 'flat'}  # No edgecolors...
        else:
            kwargs = {'shading': 'gouraud'} if shaded else {'shading': 'flat', 'edgecolors': 'k', 'linewidth': 0.1}
        tc = ax.tripcolor(v, vvalues, cmap=cmap, **kwargs)
        if zlim:
            tc.set_clim(vmin=-zlim, vmax=zlim)
    else:  # =================
        # Ok, we have a problem: tripcolor does not seem to tolerate vertex-defined colors, something we need
        # for shadows. So let's do this "manually". Internally, I think tripcolor uses a TriMesh to render
        # the mesh if gouraud is used, but somehow the later stages do not like the outcome (when doing plt.show()).
        # So, we are going to set the colors up... manually! I hate this as much as you do! ;-)
        colors = mapValues2Colors(vvalues, cmap)  # Vertex colors
        if viewkey in ['Lh-lateral', 'Lh-medial', 'L-superior']:
            normals = computeVertexNormals(vtx_L, tri_L)
        else:
            normals = computeVertexNormals(vtx_R, tri_R)

        # Create a light source object for light from
        # azimuth (from north), elevation (from 0 elevation plane). Both in degrees.
        light = LightSource(l_azimuth, l_altitude)
        shaded = lightingBias + light.shade_normals(normals) * (1-lightingBias)
        shadedColors = (colors[:,0:3].T * shaded).T

        collection = TriMesh(v)
        collection.set_facecolor(shadedColors)
        ax.add_collection(collection)
        ax.autoscale_view()
    # =================
    ax.set_aspect('equal')
    if suptitle:
        ax.set_title(suptitle, fontsize=24)
    if viewlabel:
        plt.xlabel(viewkey)


# =================================================================
# Utility functions to compute multi-views of the cortex data
# =================================================================

# plots the 6-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
#           L-flat,         R-flat
def multiview6(cortex, data, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm, suptitle='', figsize=(15, 10), **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(3, 2, 1)
    plotColorView(ax, cortex, data, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 3)
    plotColorView(ax, cortex, data, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 2)
    plotColorView(ax, cortex, data, 'Rh-lateral', cmap=rightCmap, **kwds)
    ax = plt.subplot(3, 2, 4)
    plotColorView(ax, cortex, data, 'Rh-medial', cmap=rightCmap, **kwds)

    # ================== flatmaps
    ax = fig.add_subplot(3, 2, 5)  # left hemisphere flat
    plotColorView(ax, cortex, data, 'L-flat', cmap=leftCmap, **kwds)
    ax = fig.add_subplot(3, 2, 6)  # right hemisphere flat
    plotColorView(ax, cortex, data, 'R-flat', cmap=rightCmap, **kwds)

    plt.show()


# plots a 5-view plot:
#           lh-lateral,               rh-lateral,
#                       l/r-superior,
#           lh-mdeial,                rh-medial
def multiview5(cortex, data, cmap=plt.cm.coolwarm, suptitle='', figsize=(15, 10), **kwds):
    plt.figure(figsize=figsize)
    ax = plt.subplot(2, 3, 1)
    plotColorView(ax, cortex, data, 'Lh-lateral', cmap=cmap, **kwds)
    ax = plt.subplot(2, 3, 4)
    plotColorView(ax, cortex, data, 'Lh-medial', cmap=cmap, **kwds)
    ax = plt.subplot(2, 3, 3)
    plotColorView(ax, cortex, data, 'Rh-lateral', cmap=cmap, **kwds)
    ax = plt.subplot(2, 3, 6)
    plotColorView(ax, cortex, data, 'Rh-medial', cmap=cmap, **kwds)
    ax = plt.subplot(1, 3, 2)
    plotColorView(ax, cortex, data, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(ax, cortex, data, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)
    plt.show()


# plots the 4-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
def multiview4(cortex, data, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm, suptitle='', figsize=(15, 10), **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(2, 2, 1)
    plotColorView(ax, cortex, data, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 3)
    plotColorView(ax, cortex, data, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 2)
    plotColorView(ax, cortex, data, 'Rh-medial', cmap=rightCmap, **kwds)
    ax = plt.subplot(2, 2, 4)
    plotColorView(ax, cortex, data, 'Rh-lateral', cmap=rightCmap, **kwds)

    plt.show()


# plots a left/Right-view plot:
#                       l/r-superior,
def leftRightView(cortex, data, cmap=plt.cm.coolwarm, suptitle='', figsize=(15, 10), **kwds):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    plotColorView(ax, cortex, data, 'Lh-medial', cmap=cmap, **kwds)
    ax = plt.subplot(1, 2, 2)
    plotColorView(ax, cortex, data, 'Lh-lateral', cmap=cmap, **kwds)
    plt.show()


# plots a top-view plot:
#                       l/r-superior,
def topView(cortex, data, cmap=plt.cm.coolwarm, suptitle='', figsize=(15, 10), **kwds):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    plotColorView(ax, cortex, data, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(ax, cortex, data, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)
    plt.show()
