from textattack.constraints.pre_transformation import PreTransformationConstraint
from textattack.shared.utils import default_class_repr


class MaxLengthModification(PreTransformationConstraint):
    """ 
    A constraint disallowing the modification of words which are past some maximum length limit
    """

    def __init__(self, max_length):
        self.max_length = max_length

    def _get_modifiable_indices(self, current_text):
        """ Returns the word indices in current_text which are able to be deleted """
        return set(range(min(self.max_length, len(current_text.words))))