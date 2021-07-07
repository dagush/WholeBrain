# =================================================================
# Examnples of siibra in action, to test it works.
#
# Taken from
# https://siibra-python.readthedocs.io/en/latest/usage.html
# =================================================================

# Retrieving receptor densities for one brain area
def main():
    import siibra
    # NOTE: assumes the client is already authenticated, see above
    atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
    atlas.select_region('v1')
    features = atlas.get_features(
        siibra.features.modalities.ReceptorDistribution)
    for r in features:
        fig = r.plot(r.region)


# Retrieving gene expressions for one brain area
def main2():
    import siibra
    from nilearn import plotting
    atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
    # request gene expressions from Allen Atlas
    atlas.select_region("v1 left")
    features = atlas.get_features(
        siibra.features.modalities.GeneExpression,
        gene=siibra.features.gene_names.GABARAPL2 )
    print(features[0])

    # plot
    all_coords = [tuple(g.location) for g in features]
    mask = atlas.build_mask(siibra.spaces.MNI152_2009C_NONL_ASYM)
    display = plotting.plot_roi(mask)
    display.add_markers(all_coords,marker_size=5)
    plotting.show()


if __name__ == '__main__':
    # main()
    main2()
