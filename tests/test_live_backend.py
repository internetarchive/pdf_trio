#!/usr/bin/env python3

"""
These tests attempt to use the actual docker back-end, without mocks, as an
integration test.
"""

import os
import json
import pytest
import requests

from fixtures import flask_client, skip_if_no_tensorflow


def do_classify_on_pdf(pdf_file_path, flask_client):
    """
    Helper to read in sample pdf one at a time and classify them
    """
    target_url = "/classify/research-pub/all"
    print("-----")
    print("process %s" % (pdf_file_path))
    with open(pdf_file_path, 'rb') as f:
        #filename = os.path.basename(pdf_file_path)
        filename = pdf_file_path
        form_data = {
            "pdf_content": (filename, f, "application/octet-stream")
        }
        response = flask_client.post(target_url, data=form_data)
    # print(response.headers)  # DEBUG
    assert response.status_code == 200
    print("%s  %s" % (response.data, pdf_file_path))  # DEBUG

    # check that the responses aren't default values
    assert response.json['ensemble_score'] != 0.5
    assert response.json['linear_score'] != 0.5


def collect_files(dir_path):
    ret_list = []
    for fname in os.listdir(dir_path):
        fname_full = dir_path + "/" + fname
        if os.path.isfile(fname_full):
            ret_list.append(fname_full)
    return ret_list


def test_all(flask_client):
    skip_if_no_tensorflow()
    #  ./tests/files/other and ./tests/files/research/
    negative_samples = collect_files("tests/files/other")
    print(negative_samples)
    positive_samples = collect_files("tests/files/research")
    print(positive_samples)
    for fname in negative_samples:
        do_classify_on_pdf(fname, flask_client)
    for fname in positive_samples:
        do_classify_on_pdf(fname, flask_client)
