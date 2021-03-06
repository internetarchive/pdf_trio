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

"""

"""
PDF processing.
We use pdftotext (exec'ed) because it works more often than PyPDF2.
Images from PDFs are created by ImageMagick (and ghostscript).
"""

from io import BytesIO
import time
import shutil
import subprocess
import logging
import numpy as np
from cv2 import cv2  # pip install opencv-python  to get this

log = logging.getLogger(__name__)

if not shutil.which('pdftotext'):
    print("ERROR: you do not have pdftotext installed. Install it first before calling this script")
    log.error("the required executable pdftotext is not installed")

if not shutil.which('convert'):
    print("ERROR: you do not have convert from ImageMagick installed. Install it first before calling this script")
    log.error("the required executable convert from ImageMagick which is not installed")


def extract_pdf_text(pdf_content, trace_name):
    """
    Extract text from PDF. The text is extracted in human-reading order. EOL chars are present.
    :param pdf_content: as binary string object.
    :param trace_name the filename on the client, for traceability
    :return: text string of extracted human readable text from PDF, zero length string if could not extract or no text.
    """
    text = ""
    # specify UTF-8 for output
    p_args = ['pdftotext', '-nopgbrk', '-eol', 'unix', '-enc', 'UTF-8', "-", "-"]
    t0 = time.time()
    # start subprocess, encoding not specified since input must be binary
    pp = subprocess.Popen(p_args, bufsize=262144, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # pump the content into stdin
    try:
        pp.stdin.write(pdf_content)
    except IOError:
        pass
    text_binary = b""
    # wait to finish, drain pipes
    try:
        outs, errs = pp.communicate(timeout=30)
        # outs and errs are file handles
        #  outs was read as binary, but it is actually UTF-8, so we decode here
        text_binary = outs
    except subprocess.TimeoutExpired:
        pp.kill()
        # drain residue so subprocess can really finish
        outs, errs = pp.communicate()
        text = outs  # get at least some text from file
        log.warning("pdftotext, processing for %s did not terminate in %.2f seconds, terminating." %
                    (trace_name, time.time() - t0))
    # convert from binary utf-8 string to text string
    try:
        text = text_binary.decode("utf-8")
    except UnicodeError:
        log.warning("pdftotext, processing for %s utf-8 decode exception occurred." % (trace_name))
    return text


def extract_pdf_image(pdf_content, trace_name, page=0):
    """
    ImageMagick (and ghostscript) is used to generate the image.

    Image is returned as an array of floats with shape (224, 224, 3).

    :param pdf_content: as binary string object.
    :param trace_name the filename on the client, for traceability
    :param page:  page number (from 0)
    :return: array of floats with shape (299, 299, 3), None is returned if no good image
    produced.
    """
    jpg_content = None
    pageSpec = "[" + str(page) + "]"
    # start subprocess
    # Use pipes so that stderr can be collected (and not just mixed into main process stderr, hiding other errors)

    # the parameters here must match training to maximize accuracy
    convert_cmd = ['convert', "pdf:-" + pageSpec, '-background', 'white',
                   '-alpha', 'remove', '-equalize', '-quality', '95',
                   '-thumbnail', '156x', '-gravity', 'north', '-extent',
                   '224x224', "jpg:-"]
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        log.debug("ImageMagick Command=" + " ".join(convert_cmd))
    t0 = time.time()
    pp = subprocess.Popen(convert_cmd, bufsize=262144, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    try:
        pp.stdin.write(pdf_content)
    except IOError:
        pass

    try:
        outs, errs = pp.communicate(timeout=30)
        # get jpg bytes
        jpg_content = outs
        # check if jpg sufficient size
        if len(jpg_content) <= 3000:
            jpg_content = None  # probably a blank image, so ignore it
            log.warning("ignoring blank jpg produced by imagemagick for %s" % trace_name)
    except subprocess.TimeoutExpired:
        pp.kill()
        # drain residue so subprocess can really finish
        outs, errs = pp.communicate()
        log.warning("convert command (imagemagick) on %s did not terminate in %.2f seconds, terminating." %
                    (trace_name, time.time()-t0))
    if jpg_content is None:
        return None
    # convert jpg_content to image as array
    img_stream = BytesIO(jpg_content)
    img_array = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        log.warning("imdecode failed for %s" % (trace_name))
        return None
    img_array = img_array.astype(np.float32)
    #img_array = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
    # we have 224x224, resize to 299x299 for shape (224, 224, 3)
    # ToDo: target size could vary, depending on the pre-trained model, should auto-adjust
    img299 = cv2.resize(img_array, dsize=(299, 299), interpolation=cv2.INTER_LINEAR)
    return img299
