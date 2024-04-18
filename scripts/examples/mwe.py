"""Minimum working example for superblockify."""

# While this script is running, figures will be generated on screen. 
# Close the figure windows to proceed with the script execution.


from superblockify import ResidentialPartitioner

if __name__ == "__main__":
    # --- Initialize the partitioner ---
    partitioner = ResidentialPartitioner(
        name="Scheveningen_test",
        city_name="Scheveningen",
        search_str="Scheveningen, NL",
        unit="time",  # "time", "distance", any other edge attribute, or None to count edges
    )
    # If you want to select a different city, find the corresponding search string
    # (`search_str`) at https://nominatim.openstreetmap.org/. 
    # The smaller the place, the quicker the partitioning.
    # For large places sufficient memory is required.

    # --- Run the partitioner ---
    partitioner.run(
        calculate_metrics=True,
        make_plots=True,
        replace_max_speeds=False,
    )

    # --- Save the partitioner ---
    # Save it to disk, with all attributes, without an extra copy of the
    # graph. The original graph has been cached in the data/graphs folder.
    partitioner.save()
