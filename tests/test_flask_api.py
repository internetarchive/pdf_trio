
import os
import json
import pytest
import responses

import pdf_trio
from fixtures import flask_client


def test_api_misc_routes(flask_client):
    misc_routes = [
        "/",
        "/api/list",
    ]
    for r in misc_routes:
        resp = flask_client.get(r)
        assert resp.status_code == 200


def test_api_classify_url(flask_client):
    """
    Test Classify By URL
    """
    headers = {"content-type": "application/json"}
    url_map = {"urls": [
        "https://arxiv.org/pdf/1607.01759.pdf",
        "https://example.com/maps/foo.pdf",
    ]}
    url_json = json.dumps(url_map)
    json_response = flask_client.post(
        "/classify/research-pub/url",
        data=url_json,
        headers=headers,
    )
    assert json_response.status_code == 200

    # expecting json like:   { "url1": 0.88, "url2": 0.23 }
    print("verbatim response=%s" % (json_response.data))
    predictions = json_response.get_json()["predictions"]
    for k in predictions:
        print("%.2f : %s" % (predictions[k], k))
    assert type(predictions[url_map['urls'][0]]) == float
    assert predictions[url_map['urls'][0]] != 0.5
    assert predictions[url_map['urls'][1]] != 0.5


@responses.activate
def test_api_classify_pdf(flask_client):

    test_pdf_path = 'tests/files/research/submission_363.pdf'

    tf_bert_json = {'outputs': [[0.000686553773, 0.999313474]]}
    tf_image_json = {'predictions': [[0.999999881, 1.45352288e-07]]}

    responses.add(responses.POST, 'http://localhost:8601/v1/models/bert_model:predict',
        json=tf_bert_json, status=200)
    responses.add(responses.POST, 'http://localhost:8501/v1/models/image_model:predict',
        json=tf_image_json, status=200)

    for mode in ('all', 'auto'):
        with open(test_pdf_path, 'rb') as f:
            form_data = {
                "pdf_content": (test_pdf_path, f, "application/octet-stream")
            }
            response = flask_client.post(
                "/classify/research-pub/" + mode,
                data=form_data,
            )
            assert response.status_code == 200
            assert response.json['status'] == "success"

            # check that the responses aren't default values
            assert response.json['ensemble_score'] != 0.5
            assert response.json['linear_score'] != 0.5
            #assert response.json['versions']['model_date']
            assert response.json['versions']['git_rev']
            assert response.json['versions']['pdftrio_version'] == pdf_trio.__version__

    assert len(responses.calls) == 2
