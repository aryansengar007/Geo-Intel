from backend.app.services.spectral_indices import compute_ndbi, compute_ndvi, compute_ndwi


def test_ndvi_formula_is_basic_and_numeric():
    assert compute_ndvi(10, 20) == 0.3333333333333333
    assert compute_ndvi(0, 0) == 0.0


def test_ndwi_formula_is_basic_and_numeric():
    assert compute_ndwi(20, 10) == 0.3333333333333333


def test_ndbi_formula_is_basic_and_numeric():
    assert compute_ndbi(20, 10) == 0.3333333333333333
