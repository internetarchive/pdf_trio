
import os

from pdf_trio import pdf_util
import numbers as np


def test_extract_pdf_text():

    test_pdf_path = 'tests/files/research/fea48178ffac3a42035ed27d6e2b897cb570cf13.pdf'
    with open(test_pdf_path, 'rb') as f:
        pdf_content = f.read()
        text = pdf_util.extract_pdf_text(pdf_content, test_pdf_path)

    assert text
    assert "Yoshiyuki" in text


def test_extract_pdf_image():

    test_pdf_path = 'tests/files/research/submission_363.pdf'
    with open(test_pdf_path, 'rb') as f:
        pdf_content = f.read()
        img_as_array = pdf_util.extract_pdf_image(pdf_content, test_pdf_path)
    print("img_as_array type is %s" % (type(img_as_array)))
    # shape is (299, 299, 3)
    assert img_as_array is not None
    assert img_as_array.shape[0] == 299
    assert img_as_array.shape[1] == 299
    assert img_as_array.shape[2] == 3
    #   test "no image"  ; the pdf above has only one page 
    img_as_array = pdf_util.extract_pdf_image(pdf_content, test_pdf_path, page=2)
    # shape is (299, 299, 3)
    assert img_as_array is None
