"""Tests for the abstract attribute-based partitioning approach."""

import pytest

from superblockify.partitioning.approaches.attribute import AttributePartitioner


class TestAttributePartitioner:
    """Tests for the abstract attribute-based partitioner."""

    # pylint: disable=too-few-public-methods

    def test_write_attribute_not_overwritten(self):
        """If write_attribute is not overwritten, raise a TypeError
        when instantiating the class."""
        with pytest.raises(TypeError):
            # pylint: disable=too-few-public-methods
            class NotOverwritten(AttributePartitioner):
                """Partitioner that does not overwrite write_attribute."""

            NotOverwritten()  # pylint: disable=abstract-class-instantiated
