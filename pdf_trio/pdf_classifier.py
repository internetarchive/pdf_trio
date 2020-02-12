#!/usr/bin/env python3

"""
Copyright 2019 Internet Archive

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Inference drivers for CNN, FastText, and BERT for PDF classification here.
"""

import os
import time
import json
import logging
import argparse
import subprocess

from cv2 import cv2  # pip install opencv-python  to get this
import numpy as np
import requests
import fasttext
from fasttext import load_model

from pdf_trio import text_prep
from pdf_trio import pdf_util



log = logging.getLogger(__name__)


class PdfClassifier:


    def __init__(self, **kwargs):

        self.version_map = {}
        self.json_content_header = {"Content-Type": "application/json"}

        image_server_prefix = os.environ.get('TF_IMAGE_SERVER_URL')
        if not image_server_prefix:
            raise ValueError('Missing TF image classifier URL config, ' +
                'define env var TF_IMAGE_SERVER_URL')
        self.image_tf_server_url = image_server_prefix + "/models/image_model"
        # self.version_map['image'] is lazy-loaded

        bert_server_prefix = os.environ.get('TF_BERT_SERVER_URL')
        if not bert_server_prefix :
            raise ValueError('Missing TF BERT classifier URL config, ' +
                'define env var TF_BERT_SERVER_URL')
        self.bert_tf_server_url = bert_server_prefix + "/models/bert_model"
        # self.version_map['bert'] is lazy-loaded

        vocab_path = os.environ.get('TF_BERT_VOCAB_PATH')
        if not vocab_path:
            raise ValueError('TF_BERT_VOCAB_PATH is not set to the path to vocab.txt')
        if not os.path.exists(vocab_path) and os.path.isfile(vocab_path):
            raise ValueError('TF_BERT_VOCAB_PATH target does not exist: %s' % vocab_path)
        log.warning("Loading BERT model vocabulary...")
        self.bert_vocab = text_prep.load_bert_vocab(vocab_path)

        model_path = os.environ.get('FT_MODEL')
        if not model_path:
            raise ValueError('Missing fasttext model, ' +
                'define env var FT_MODEL=full_path_to_basename')
        log.warning("Loading fasttext model...")
        self.fasttext_model = fasttext.load_model(model_path)
        self.version_map["linear_model"] = os.environ.get('FT_MODEL_VERSION') or None

        self.version_map["models_date"] = os.environ.get('PDFTRIO_MODELS_DATE') or None

    @staticmethod
    def get_tf_model_version(url):
        """
        Connect to back-end tensorflow-serving APIs and fetch model version
        metadata
        """
        resp = requests.get(url)
        resp.raise_for_status()
        status = resp.json()['model_version_status']
        assert status['state'] == "AVAILABLE"
        return status['version']

    def lazy_load_versions(self):
        """
        Want to move these back-end connections out of __init__()
        """
        if not 'image_model' in self.version_map:
            self.version_map['image_model'] = self.get_tf_model_version(self.image_tf_server_url)
        if not 'bert_model' in self.version_map:
            self.version_map['bert_model'] = self.get_tf_model_version(self.bert_tf_server_url)

    def classify_pdf_multi(self, modes, pdf_filestorage):
        """
        Use the modes param to pick subclassifiers and make an ensemble conclusion.

        example result:

            {
                "ensemble_score" : 0.94,
                "image_score" : 0.96,
                "linear_score" : 0.92,
                "bert_score" : 0.91,
                "versions" : {
                    "pdftrio_version": "0.1.0",
                    "models_date": "2020-01-15",
                    "git_rev": "2b1845b0",
                    "image_model" : "20190708",
                    "linear_model" : "20190720",
                    "bert_model" : "20190807",
                }
            }

        :param modes:  comma sep list of 1 or more {auto, image, linear, bert, all}
        :param pdf_filestorage: as FileStorage object (contains a stream).
        :return: map
        """

        self.lazy_load_versions()
        results = {"versions": self.version_map}
        confidence_values = []
        mode_list = modes.split(",")
        pdf_token_list = []
        # rewrite mode_list if 'all' is requested
        if 'all' in mode_list:
            mode_list = ['image', 'linear', 'bert']
        # write pdf content to tmp file
        tmp_pdf_name = pdf_util.tmp_file_name()
        pdf_filestorage.save(tmp_pdf_name)
        log.debug("stored pdf_content for %s in %s" % (pdf_filestorage.filename, tmp_pdf_name))
        # look ahead to see if text is required, so we can extract that now
        if ('linear' in mode_list) or ('bert' in mode_list) or ('auto' in mode_list):
            # extract text
            pdf_raw_text = pdf_util.extract_pdf_text(tmp_pdf_name)
            if len(pdf_raw_text) < 300:
                pdf_token_list = []  # too short to be useful
            else:
                pdf_token_list = text_prep.extract_tokens(pdf_raw_text)
        if 'auto' in mode_list:
            # start with fastest, use confidence thresholds to short circuit
            if len(pdf_token_list) != 0:
                # FastText
                confidence_linear = self.classify_pdf_linear(pdf_token_list)
                results['linear_score'] = confidence_linear
                confidence_values.append(confidence_linear)
                if .85 >= confidence_linear >= 0.15:
                    # also check BERT
                    pdf_token_list_trimmed = text_prep.trim_tokens(pdf_token_list, 512)
                    confidence_bert = self.classify_pdf_bert(pdf_token_list_trimmed)
                    results['bert_score'] = confidence_bert
                    confidence_values.append(confidence_bert)
            else:
                # no tokens, so use image
                jpg_file_page0 = pdf_util.extract_pdf_image(tmp_pdf_name)
                # classify pdf_image_page0
                confidence_image = self.classify_pdf_image(jpg_file_page0)
                results['image_score'] = confidence_image
                confidence_values.append(confidence_image)
                # remove tmp jpg
                if logging.getLogger().getEffectiveLevel() != logging.DEBUG:
                    pdf_util.remove_tmp_file(jpg_file_page0)
        else:
            # apply named classifiers
            for classifier in mode_list:
                if classifier == "image":
                    jpg_file_page0 = pdf_util.extract_pdf_image(tmp_pdf_name)
                    if len(jpg_file_page0) == 0:
                        log.debug("no jpg for %s" % (pdf_filestorage.filename))
                        continue  # skip
                    # classify pdf_image_page0
                    confidence_image = self.classify_pdf_image(jpg_file_page0)
                    results['image_score'] = confidence_image
                    confidence_values.append(confidence_image)
                    # remove tmp jpg
                    if logging.getLogger().getEffectiveLevel() != logging.DEBUG:
                        pdf_util.remove_tmp_file(jpg_file_page0)
                elif classifier == "linear":
                    if len(pdf_token_list) == 0:
                        # cannot use this classifier if no tokens extracted
                        log.debug("no tokens extracted for %s" % (pdf_filestorage.filename))
                        continue  # skip
                    confidence_linear = self.classify_pdf_linear(pdf_token_list)
                    results['linear_score'] = confidence_linear
                    confidence_values.append(confidence_linear)
                elif classifier == "bert":
                    if len(pdf_token_list) == 0:
                        # cannot use this classifier if no tokens extracted
                        log.debug("no tokens extracted for %s" % (pdf_filestorage.filename))
                        continue  # skip
                    pdf_token_list_trimmed = text_prep.trim_tokens(pdf_token_list, 512)
                    confidence_bert = self.classify_pdf_bert(pdf_token_list_trimmed)
                    results['bert_score'] = confidence_bert
                    confidence_values.append(confidence_bert)
                else:
                    log.warning("ignoring unknown classifier ref: " + classifier)
        if logging.getLogger().getEffectiveLevel() != logging.DEBUG:
            pdf_util.remove_tmp_file(tmp_pdf_name)
        #  compute 'ensemble_score ' using confidence_values
        if len(confidence_values) != 0:
            confidence_overall = sum(confidence_values) / len(confidence_values)
            # insert confidence_overall
            results['ensemble_score'] = confidence_overall
        return results


    @staticmethod
    def encode_confidence(label, confidence):
        """
        Encode label,confidence into a single float [0.0, 1.0] so that 1.0 means perfect confidence in positive case,
        and 0.0 is perfect confidence for negative case.
        confidence 0 means 'other', 1.0 means 'research'
        :param label: 'research' or '__label__research' versus 'other' or '__label__other'
        :param confidence: [0.5, 1.0]
        :return: [0.0, 1.0]
        """
        if confidence < 0.5:
            log.error("encode_confidence called improperly with label=%s confidence=%f" % (label, confidence))
        if confidence > 1.0:
            confidence = 1.0
        if confidence < 0.0:
            confidence = 0.0
        if label == '__label__research' or label == 'research':
            return (confidence / 2) + 0.5
        return 0.5 - (confidence / 2)


    @staticmethod
    def decode_confidence(e):
        """
        decode composite confidence
        :param e: combined/encoded confidence
        :return: label, confidence
        """
        if e < 0.5:
            return "other", 1.0 - (2 * e)
        return "research", (2 * e) - 1.0


    def classify_pdf_linear(self, pdf_token_list):
        """
        Apply fastText model to content

        :param pdf_tokens: cleaned tokens list from pdf content
        :return: encoded confidence as type float with range [0.5,1.0] that example is positive
        """
        #  classify using fastText model
        results = self.fasttext_model.predict(" ".join(pdf_token_list))
        label = results[0][0]
        confidence = results[1][0]
        log.debug("classify_pdf_linear: label=%s confidence=%.2f" % (label, confidence))
        return self.encode_confidence(label, confidence)


    def classify_pdf_bert(self, pdf_token_list, trace_id=""):
        """
        Apply BERT model to content to classify given token list.

        :param pdf_tokens: cleaned tokens list from pdf content, trimmed to not exceed max tokens (512)
        :param trace_id: string for doc id, if known.
        :return: encoded confidence as type float with range [0.5,1.0] that example is positive
        """
        token_ids = text_prep.convert_to_bert_vocab(self.bert_vocab, pdf_token_list)
        tcount = len(token_ids)
        if tcount < 512:
            for j in range(tcount, 512):
                token_ids.append(0)
        # add entries so that token_ids is 512 length
        # for REST request, need examples=[{"input_ids": [], "input_mask":[], "label_ids":[0], "segment_ids":[]}]
        input_ids = token_ids
        if tcount < 512:
            input_mask = np.concatenate((np.ones(tcount, dtype=int), np.zeros(512-tcount, dtype=int)), axis=0).tolist()
        else:
            input_mask = np.ones(512, dtype=int).tolist()
        label_ids = [0]  # dummy, one int, not needed for prediction
        segment_ids = np.zeros(512, dtype=int).tolist()
        # The released BERT graph has been tweaked to use 4 input placeholders,
        #   so we use "inputs" columnar format REST style.
        #   Columnar format means each named input can have a list of values, but we actually are only submitting
        #   one at a time here.
        #   label_ids is a scalar (placeholder shape [None]).
        evalue = {"input_ids": [input_ids],
                "input_mask": [input_mask],
                "label_ids": label_ids,
                "segment_ids": [segment_ids]}
        req_json = json.dumps({"signature_name": "serving_default", "inputs":  evalue})
        log.debug("BERT: request to %s is: %s ... %s" % (self.bert_tf_server_url + ":predict", req_json[:80], req_json[len(req_json)-50:]))
        ret = 0.5  # zero confidence encoded default
        try:
            response = requests.post(self.bert_tf_server_url + ":predict", data=req_json, headers=self.json_content_header)
            if response.status_code == 200:
                response_vec = response.json()["outputs"][0]
                confidence_other = response_vec[0]
                confidence_research = response_vec[1]
                log.debug("bert classify %s  other=%.2f research=%.2f" % (trace_id, confidence_other, confidence_research))
                if confidence_research > confidence_other:
                    ret = self.encode_confidence("research", confidence_research)
                else:
                    ret = self.encode_confidence("other", confidence_other)
            elif response.status_code == 400:
                log.warning("HTTP 400 from tf-serving of bert, %s" % response.json())
        except Exception:
            log.warning("exception occurred processing REST BERT tensorflow-serving for %s" % trace_id)
        return ret



    def classify_pdf_image_via_exec(self, jpg_file):
        """
        Apply image model to content image

        :param jpg_file: tmp jpg image file name, full path.
        :return: encoded confidence as type float with range [0.5,1.0] that example is positive
        """
        ret = 0.5  # lowest confidence encoded value
        myenv = {"dummy": "0",
                "CONDA_DEFAULT_ENV": "tf_hub",
                "CONDA_PYTHON_EXE": "/home/peb/miniconda3/bin/python",
                "HOME": "/home/peb",
                "PATH": "/home/peb/bin:/home/peb/miniconda3/envs/tf_hub/bin:/home/peb/miniconda3/bin:/home/peb/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "CONDA_EXE": "/home/peb/miniconda3/bin/conda",
                "CONDA_PREFIX": "/home/peb/miniconda3/envs/tf_hub",
                "CONDA_PREFIX_1": "/home/peb/miniconda3",
                "CONDA_SHLVL":"2"
                }
        w_dir = "../tf_hub_image_classifier"
        cmd = [w_dir+'/infer_image_new.py', '--image='+jpg_file, '--graph='+w_dir+'/retrained_graph.pb',
                '--labels='+w_dir+'/out/retrained_labels.txt', '--input_layer=Placeholder', '--output_layer=final_result']
        mycwd = "."
        t0 = time.time()
        pp = subprocess.Popen(cmd, encoding='utf-8', env=myenv, cwd=mycwd, bufsize=1, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = pp.communicate(timeout=30)
            # dump stderr if DEBUG
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                log.debug("infer_image_new.py stderr: %s", errs)
            # parse outs     ex: research 0.5082055 ./tmp/0b4997a068e557eb92b6adf7875248ec7292dd4a.jpg
            lines = outs.split('\n')
            for aline in lines:
                mytokens = aline.split()
                if len(mytokens) < 3:
                    continue
                label = mytokens[0]
                if label == "research" or label == "other":
                    confidence = float(mytokens[1])
                    if confidence > 0.5:
                        ret = self.encode_confidence(label, confidence)
                        break
        except subprocess.TimeoutExpired:
            pp.kill()
            # drain residue so subprocess can really finish
            outs, errs = pp.communicate()
            log.warning("classify_pdf_image, command did not terminate in %.2f seconds, terminating." % (time.time()-t0))


        #log.info("classify_pdf_image: label=%s confidence=%.2f" % (label, confidence))
        return ret


    def classify_pdf_image(self, jpg_file):
        """
        Apply image model to content image using tensorflow-serving.

        :param jpg_file: tmp jpg image file name, full path.
        :return: encoded confidence as type float with range [0.5,1.0] that example is positive
        """
        ret = 0.5  # lowest confidence encoded value
        img = cv2.imread(jpg_file).astype(np.float32)
        # we have 224x224, resize to 299x299 for shape (224, 224, 3)
        # ToDo: target size could vary, depending on the pre-trained model, should auto-adjust
        img299 = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_LINEAR)
        my_images = np.reshape(img299, (-1, 299, 299, 3))
        req_json = json.dumps({"signature_name": "serving_default", "instances": my_images.tolist()})
        try:
            response = requests.post(self.image_tf_server_url + ":predict", data=req_json, headers=self.json_content_header)
            if response.status_code == 200:
                response_vec = response.json()["predictions"][0]
                confidence_other = response_vec[0]
                confidence_research = response_vec[1]
                log.debug("image classify %s  other=%.2f research=%.2f" % (jpg_file, confidence_other, confidence_research))
                if confidence_research > confidence_other:
                    ret = self.encode_confidence("research", confidence_research)
                else:
                    ret = self.encode_confidence("other", confidence_other)
        except Exception:
            log.warning("exception occurred processing REST tensorflow-serving for %s" % jpg_file)
        return ret


def main():
    """
    Apply the preprocessing to generate
    """
    #  ToDo: process PDF files using all classifiers and create various confusion matrices to see where
    #     models agree/disagree. Data comes from directories associated with a defined category.
    #     REST api is not used.
    # init the arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_positive", type=str, default='',
                        help="dir. with positive PDF files to process, use a comma sep for multiple dirs",
                        required=True)
    parser.add_argument("--input_negative", type=str, default='',
                        help="dir. with negative PDF files to process, use a comma sep for multiple dirs",
                        required=True)
    parser.add_argument("--temp", type=str, default='/tmp', help="temp dir to use during extraction, /tmp default",
                        required=False)
    parser.add_argument("--skip", type=str, default='',
                        help="file containing file basenames to skip, one per line, optional", required=False)
    parser.add_argument("--testing", default=False, help="testing mode, be verbose, do only a few cycles",
                        action="store_true")

    # read arguments from the command line
    args = parser.parse_args()

    # TODO: doesn't actually call anything

if __name__ == '__main__':
    main()
