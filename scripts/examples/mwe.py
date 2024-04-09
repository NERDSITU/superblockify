"""Minimum working example for superblockify."""

from superblockify import ResidentialPartitioner


if __name__ == "__main__":
    part = ResidentialPartitioner(
        name="Salerno_mwe",
        city_name="Salerno",
        search_str="Salern, Campania, Italy",
        # reload_graph=True
    )
    # Find search strings via https://nominatim.openstreetmap.org/
    # The smaller the place, the quicker the partitioning
    # For large places sufficient memory is required

    # Save the preprocessed graph to disk:
    # >>> part.save(save_graph_copy=True)
    # If loading a previously saved graph, all previous steps are not needed:
    # >>> part = ResidentialPartitioner.load("Salerno_mwe")

    part.run(
        calculate_metrics=True,
        make_plots=True,
        replace_max_speeds=False,
    )

    # Save the partitioner to disk, with all attributes, without an extra copy of the
    # graph. The original graph has been cached in the data/graphs folder.
    part.save()

# or
# >>> import superblockify as sb
# >>> part = sb.ResidentialPartitioner(...)
# >>> part.run(...)
# >>> sb.save_to_gpkg(part)

# If you have the whole GHS POP raster, set the path after importing like so:
# >>> sb.config.Config.FULL_RASTER = ...
