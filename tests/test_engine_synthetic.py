import pytest
from config import EXPECTED_ID2LABEL
from engine import validate_score_vector

LABELS=tuple(EXPECTED_ID2LABEL[i] for i in sorted(EXPECTED_ID2LABEL))
def test_score_validation_accepts_valid_distribution():
    values=[{'label':label,'score':1/7} for label in LABELS]
    assert len(validate_score_vector(values,LABELS,7))==7

def test_score_validation_rejects_invalid_distribution():
    values=[{'label':label,'score':.2} for label in LABELS]
    with pytest.raises(Exception): validate_score_vector(values,LABELS,7)
