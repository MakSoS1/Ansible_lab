import pytest
from ecup_matching.v15_train_config import get_variant

def test_v15_causal_ladder():
    a0=get_variant('A0'); a1=get_variant('A1'); a2=get_variant('A2'); a3=get_variant('A3'); a4=get_variant('A4')
    assert (a0.include_attributes,a0.use_typed_features,a0.use_category_head,a0.macro_balanced)==(False,False,False,False)
    assert (a1.include_attributes,a1.use_typed_features,a1.use_category_head,a1.macro_balanced)==(True,False,False,False)
    assert (a2.include_attributes,a2.use_typed_features,a2.use_category_head,a2.macro_balanced)==(True,True,False,False)
    assert (a3.include_attributes,a3.use_typed_features,a3.use_category_head,a3.macro_balanced)==(True,True,True,False)
    assert (a4.include_attributes,a4.use_typed_features,a4.use_category_head,a4.macro_balanced)==(True,True,True,True)
    assert all(get_variant(x).max_length==128 for x in ('A0','A1','A2','A3','A4'))

def test_unknown_variant_fails_closed():
    with pytest.raises(ValueError): get_variant('A5')
