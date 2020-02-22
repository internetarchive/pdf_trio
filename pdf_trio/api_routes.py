
"""
Copyright 2020 Internet Archive

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import logging

from flask import request, jsonify, abort, Blueprint, current_app
from pdf_trio import pdf_classifier, url_classifier


logging.basicConfig(filename='research-pub.log', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.info("STARTING   STARTING   STARTING   STARTING   STARTING   STARTING   STARTING")

bp = Blueprint("classify", __name__)

bp.pdf_classifier = pdf_classifier.PdfClassifier()
bp.url_classifier = url_classifier.UrlClassifier()

@bp.route('/classify/research-pub/url', methods = ['POST'])
def classify_by_url():
    """
    The given URL(s) is/are classified as referring to a research publication with a confidence value.
    A PDF under arxiv.org has a high confidence of being a research work.
    A PDF under amazon.com has a low confidence of being a research work.
    Parameter urls= in POST: json list of URLs like ["http://foo.com", "http://bar.com"]
    Each URL is separately classified.
    :return: json { "url1": 0.88, "url2": 0.92, "url3": 0.23 }
    """
    start_ts = int(time.time()*1000)
    input = request.json or {}
    url_list = input.get('urls')
    results_map = {}
    for url in url_list:
        confidence = bp.url_classifier.classify_url(url)
        results_map[url] = confidence
    log.debug("results_map=%s" % (results_map))
    retmap = {"predictions": results_map}
    return jsonify(retmap)

@bp.route('/classify/research-pub/<string:ctype>', methods = ['POST'])
def classify_pdf(ctype):
    """
    The given PDF is classified as a research publication or not. The PDF is not stored.
    params: "type" comma sep. list of { all, auto, image, bert, linear }
    :return: json

    Example result:

        {
          "status": "success",
          "ensemble_score" : 0.94,
          "image_score" : 0.96,
          "linear_score" : 0.92,
          "bert_score" : 0.91,
          "versions" : {
            "models_date": "2020-01-15",
            "image_model" : "20190708",
            "linear_model" : "20190720",
            "bert_model" : "20190807",
          }
        }
    """
    #log.debug("Request headers: %s" % (request.headers))
    log.debug("ctype=%s" % (ctype))
    if ctype is None:
        ctype = "auto"
    pdf_filestorage = request.files['pdf_content']
    filename = pdf_filestorage.filename
    pdf_stream = pdf_filestorage.stream
    pdf_content = pdf_stream.read()
    log.debug("type=%s  pdf_content for %s with length %d" % (ctype, filename, len(pdf_content)))
    results = bp.pdf_classifier.classify_pdf_multi(ctype, pdf_content, filename)
    if current_app.config['GIT_REV']:
        results['versions']['git_rev'] = current_app.config['GIT_REV']
    if current_app.config['VERSION']:
        results['versions']['pdftrio_version'] = current_app.config['VERSION']
    if 'ensemble_score' in results:
        results['status'] = "success"
        return jsonify(results), 200
    else:
        results['status'] = "error"
        return jsonify(results), 400

