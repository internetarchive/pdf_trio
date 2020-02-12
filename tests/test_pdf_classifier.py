
import os
import json
import pytest
import requests
import responses
from werkzeug.datastructures import FileStorage

from pdf_trio.pdf_classifier import PdfClassifier


@responses.activate
def test_pdf_classifier_basics():
    c = PdfClassifier()

    test_pdf_path = 'tests/files/research/submission_363.pdf'

    tf_bert_json = {'outputs': [[0.000686553773, 0.999313474]]}
    tf_image_json = {'predictions': [[0.999999881, 1.45352288e-07]]}

    responses.add(responses.POST, 'http://localhost:8601/v1/models/bert_model:predict',
        json=tf_bert_json, status=200)
    responses.add(responses.POST, 'http://localhost:8501/v1/models/image_model:predict',
        json=tf_image_json, status=200)

    for mode in ("linear", "bert", "image"):
        with open(test_pdf_path, 'rb') as f:
            pdf_fs = FileStorage(f)
            resp = c.classify_pdf_multi(mode, pdf_fs)
            print(resp)
            print(PdfClassifier.decode_confidence(resp[mode]))
            assert resp[mode] != 0.5
            assert resp['is_research'] != 0.5

    assert len(responses.calls) == 2

    for mode in ("auto", "all"):
        with open(test_pdf_path, 'rb') as f:
            pdf_fs = FileStorage(f)
            resp = c.classify_pdf_multi(mode, pdf_fs)
            assert resp['is_research'] != 0.5

    assert len(responses.calls) == 4
