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


Text tokenization helpers.
"""

import collections
import re
import logging
import string

log = logging.getLogger("text_prep")

# define punctuation chars !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~  plus some unicode punct ¿؟¡¢
expanded_punct = string.punctuation + chr(191) + chr(1567) + chr(161) + chr(162)
# £¥©®§
expanded_punct = expanded_punct + chr(163) + chr(165) + chr(169) + chr(174) + chr(167)
#  unicode additions (found with https://unicode-table.com/en/blocks/general-punctuation/)
#     this picks up en-dash, em-dash and punctuation seen in (at least) Russian
for i in range(8192, 8303):
    if i == 8293:  # unused
        continue
    expanded_punct = expanded_punct + chr(i)

def extract_tokens(str_content):
    """
    Clean given string and return the cleaned list of tokens.
    Punctuation, excess whitespace, control chars, funny stuff, is removed.
    :param str_content: raw string
    :return: token list, preserving order from document.
    """
    # convert control and EOL chars to space
    str_content = re.sub(r'[\x00-\x1F]+', ' ', str_content)
    # split by whitespace
    tokens = str_content.split()

    ttable = str.maketrans('', '', expanded_punct)
    tokens = [w.translate(ttable) for w in tokens]
    return tokens


def trim_tokens(file_token_list, max_tokens):
    """
    If given list has more than given max_tokens, return an even split between the head and tail of the
    list so that total number of tokens is max_tokens.
    :param file_token_list: the tokens to check.
    :param max_tokens: max number of tokens to return as a list.
    :return:
    """
    ntokens = len(file_token_list)
    if ntokens > max_tokens:
        cut_out = ntokens - max_tokens
        front_end_offset = int(max_tokens/2)
        back_begin_offset = front_end_offset + cut_out
        file_token_list = file_token_list[:front_end_offset] + file_token_list[back_begin_offset:]
    return file_token_list


def load_bert_vocab(vocab_file):
    """
    Loads a vocabulary file into a dictionary.
    :param vocab_file:  path to vocab.txt from pre-trained BERT model
    :return: dictionary.
    """
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as file:
        while True:
            token = file.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_to_bert_vocab(vocab, items):
    """
    Converts a sequence of [tokens|ids] using the vocab.
    Tokens not in dictionary are skipped.
    :param vocab: dictionary
    :param items: list of tokens (strings)
    :return:
    """
    output = []
    for item in items:
        try:
            output.append(vocab[item])
        except KeyError:
            continue
    return output
