"""Minimum working example for superblockify."""
from superblockify import ResidentialPartitioner

if __name__ == "__main__":
    part = ResidentialPartitioner(
        name="Milan_mwe", city_name="Milan", search_str="Milan, Lombardy, Italy"
    )
    part.run(make_plots=True)
    part.calculate_metrics(make_plots=True, num_workers=48, chunk_size=2)
    part.save()  # Save the partitioner to disk
