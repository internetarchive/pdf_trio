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

import os
import time
import pathlib
import time
import shutil
import subprocess
import random
import datetime
import atexit
import logging

log = logging.getLogger(__name__)

if not shutil.which('pdftotext'):
    print("ERROR: you do not have pdftotext installed. Install it first before calling this script")
    log.error("the required executable pdftotext is not installed")

if not shutil.which('convert'):
    print("ERROR: you do not have convert from ImageMagick installed. Install it first before calling this script")
    log.error("the required executable convert from ImageMagick which is not installed")

TEMP = os.environ.get('TEMP')
if TEMP is None:
    TEMP = "/tmp"

start_datetime = datetime.datetime.now()
start_timestamp = start_datetime.isoformat().split('.')[0]
start_timestamp = start_timestamp.replace(":", "").replace("-", "")
tmp_area = TEMP + "/research-pub-area_" + start_timestamp
tmp_path = pathlib.Path(tmp_area)
tmp_path.mkdir(parents=True, exist_ok=True)


def exit_handler():
    shutil.rmtree(tmp_area, ignore_errors=True)


atexit.register(exit_handler)  # remove tmp_area on exit


def tmp_file_name(prefix="f", suffix=".pdf"):
    return str(tmp_path) + "/" + str(prefix) + str(random.randint(1, 1000000000)) + suffix


def write_tmp_file(content):
    """
    :param content: byte string to write into a temp file
    :return: full path to the file
    """
    name = tmp_file_name()
    with open(name, 'wb') as f:
        f.write(content)
    return name


def remove_tmp_file(name):
    if os.path.exists(name):
        os.remove(name)


def extract_pdf_text(pdf_tmp_file):
    """
    Extract text from PDF. The text is extracted in human-reading order. EOL chars are present.
    :param pdf_tmp_file: path to (temp) pdf file.
    :return: string of extracted human readable text from PDF, zero length string if could not extract or no text.
    """
    txt_name = pdf_tmp_file + ".txt"
    # start subprocess
    p_args = ['pdftotext', '-nopgbrk', '-eol', 'unix', '-enc', 'UTF-8', pdf_tmp_file, txt_name]
    t0 = time.time()
    pp = subprocess.Popen(p_args, encoding='utf-8', bufsize=1, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, errs = pp.communicate(timeout=30)
        # outs and errs are file handles
    except subprocess.TimeoutExpired:
        pp.kill()
        # drain residue so subprocess can really finish
        outs, errs = pp.communicate()
        log.warning("pdftotext, command did not terminate in %.2f seconds, terminating." % (time.time() - t0))
    # get text from file
    with open(txt_name, 'r', encoding='utf-8') as f:
        text = f.read()
    remove_tmp_file(txt_name)
    return text


def extract_pdf_image(pdf_tmp_file, page=0):
    """
    ImageMagick (and ghostscript) is used to generate the image.

    Caller is responsible for removing the jpg.

    :param pdf_tmp_file: path to temp file holding pdf content.
    :param page:  page number (from 0)
    :return: filename of jpg image in temporary area, caller should remove it
    when done to avoid accumulation, None is returned if no good image
    produced.
    """
    jpg_name = pdf_tmp_file + ".jpg"
    pageSpec = "[" + str(page) + "]"
    # start subprocess
    # Use pipes so that stderr can be collected (and not just mixed into main process stderr, hiding other errors)

    # the parameters here must match training to maximize accuracy
    convert_cmd = ['convert', pdf_tmp_file + pageSpec, '-background', 'white',
                   '-alpha', 'remove', '-equalize', '-quality', '95',
                   '-thumbnail', '156x', '-gravity', 'north', '-extent',
                   '224x224', jpg_name]
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        log.debug("ImageMagick Command=" + " ".join(convert_cmd))
    t0 = time.time()
    pp = subprocess.Popen(convert_cmd, encoding='utf-8', bufsize=1, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, errs = pp.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        pp.kill()
        # drain residue so subprocess can really finish
        outs, errs = pp.communicate()
        log.warning("convert command (imagemagick) on %s did not terminate in %.2f seconds, terminating." %
                    (pdf_tmp_file, time.time()-t0))
    # check if jpg file exists and sufficient size
    if os.path.exists(jpg_name):
        if os.path.getsize(jpg_name) <= 3000:
            # jpg too small, most likely blank
            log.debug("jpg too small for %s, so assumed to be a blank page; removing" % pdf_tmp_file)
            remove_tmp_file(jpg_name)
            return None
        else:
            # jpg file exists, sufficient size
            return jpg_name
    else:
        # no jpg file was produced
        log.warning("no jpg produced by imagemagick for %s" % pdf_tmp_file)
        return None
