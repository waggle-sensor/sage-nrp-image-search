"""Sagebench benchmark dataset implementation."""

from imsearch_eval.adapters.huggingface import HuggingFaceDataset


class Sagebench(HuggingFaceDataset):
    """Benchmark dataset class for Sagebench (metadata-aware image retrieval)."""

    def get_query_column(self) -> str:
        return "query_text"

    def get_query_id_column(self) -> str:
        return "query_id"

    def get_relevance_column(self) -> str:
        return "relevance_label"

    def get_metadata_columns(self) -> list:
        # These columns are included in evaluation outputs but do not control retrieval directly.
        return [
            "vsn",
            "zone",
            "host",
            "job",
            "plugin",
            "camera",
            "project",
            "address",
            "viewpoint",
            "lighting",
            "environment_type",
            "sky_condition",
            "horizon_present",
            "ground_present",
            "sky_dominates",
            "vegetation_present",
            "water_present",
            "buildings_present",
            "vehicle_present",
            "person_present",
            "animal_present",
            "night_scene",
            "precipitation_visible",
            "multiple_objects",
        ]

