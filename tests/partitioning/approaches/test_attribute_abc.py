"""Tests for the abstract attribute-based partitioning approach."""

import pytest

from superblockify.partitioning.approaches.attribute import AttributePartitioner


class TestAttributePartitioner:
    """Tests for the abstract attribute-based partitioner."""

    @pytest.mark.parametrize("attribute", [None, 1, 1.0, True, False, [], {}])
    def test__init_subclass__faulty_attribute(self, attribute):
        """If attribute is not a string, raise a ValueError."""
        with pytest.raises(ValueError):
            # pylint: disable=unused-variable, too-few-public-methods
            class FaultyAttribute(AttributePartitioner, attribute=attribute):
                """Faulty attribute partitioner."""

    def test_write_attribute_not_overwritten(self):
        """If write_attribute is not overwritten, raise a TypeError
        when instantiating the class."""
        with pytest.raises(TypeError):
            # pylint: disable=too-few-public-methods
            class NotOverwritten(AttributePartitioner, attribute="test"):
                """Partitioner that does not overwrite write_attribute."""

            NotOverwritten()  # pylint: disable=abstract-class-instantiated
