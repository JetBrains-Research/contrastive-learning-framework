# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#


import argparse
import os
from pathlib import Path

import torch
from codegen_sources.model.src.data.dictionary import (
    Dictionary,
    BOS_WORD,
    EOS_WORD,
    PAD_WORD,
    UNK_WORD,
    MASK_WORD,
)
from codegen_sources.model.src.logger import create_logger
from codegen_sources.model.src.model import build_model
from codegen_sources.model.src.utils import AttrDict
from codegen_sources.preprocessing.bpe_modes.fast_bpe_mode import FastBPEMode
from codegen_sources.preprocessing.bpe_modes.roberta_bpe_mode import RobertaBPEMode
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor

SUPPORTED_LANGUAGES = ["cpp", "java", "python"]

logger = create_logger(None, 0)


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # model
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument(
        "--BPE_path",
        type=str,
        default="data/bpe/cpp-java-python/codes",
        help="Path to BPE codes."
    )
    parser.add_argument("--code_path", type=str, default=None, help="Path to the program file")

    return parser


class Encoder:
    def __init__(self, model_path, BPE_path):
        # reload model
        reloaded = torch.load(model_path, map_location="cpu")
        # change params of the reloaded model so that it will
        # relaod its own weights and not the MLM or DOBF pretrained model
        reloaded["params"]["reload_model"] = ",".join([model_path] * 2)
        reloaded["params"]["lgs_mapping"] = ""
        reloaded["params"]["reload_encoder_for_decoder"] = False
        self.reloaded_params = AttrDict(reloaded["params"])

        # build dictionary / update parameters
        self.dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        assert self.reloaded_params.n_words == len(self.dico)
        assert self.reloaded_params.bos_index == self.dico.index(BOS_WORD)
        assert self.reloaded_params.eos_index == self.dico.index(EOS_WORD)
        assert self.reloaded_params.pad_index == self.dico.index(PAD_WORD)
        assert self.reloaded_params.unk_index == self.dico.index(UNK_WORD)
        assert self.reloaded_params.mask_index == self.dico.index(MASK_WORD)

        # build model / reload weights (in the build_model method)
        encoder, _ = build_model(self.reloaded_params, self.dico)
        self.encoder = encoder[0]
        self.encoder.cuda()
        self.encoder.eval()

        # reload bpe
        if getattr(self.reloaded_params, "roberta_mode", False):
            self.bpe_model = RobertaBPEMode()
        else:
            self.bpe_model = FastBPEMode(
                codes=os.path.abspath(BPE_path), vocab_path=None
            )

    def embed(
        self,
        input_,
        lang="cpp",
        suffix="_sa",
        device="cuda:0",
    ):
        src_lang_processor = LangProcessor.processors[lang](root_folder="tree-sitter")
        tokenizer = src_lang_processor.tokenize_code

        lang += suffix

        assert (
            lang in self.reloaded_params.lang2id.keys()
        ), f"{lang} should be in {self.reloaded_params.lang2id.keys()}"

        with torch.no_grad():
            lang_id = self.reloaded_params.lang2id[lang]

            # Convert source code to ids
            tokens = [t for t in tokenizer(input_)]
            tokens = self.bpe_model.apply_bpe(" ".join(tokens)).split()
            tokens = ["</s>"] + tokens + ["</s>"]
            input_ = " ".join(tokens)

            # Create torch batch
            len_ = len(input_.split())
            len_ = torch.LongTensor(1).fill_(len_).to(device)
            x = torch.LongTensor([self.dico.index(w) for w in input_.split()]).to(device)[:, None]
            langs = x.clone().fill_(lang_id)

            # Encode
            enc_ = self.encoder("fwd", x=x, lengths=len_, langs=langs, causal=False)
            enc_ = enc_.transpose(0, 1)

            return enc_


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    assert os.path.isfile(params.model_path), f"The path to the model checkpoint is incorrect: {params.model_path}"
    assert os.path.isfile(params.BPE_path), f"The path to the BPE tokens is incorrect: {params.BPE_path}"
    assert os.path.isfile(params.code_path), f"The path to the code is incorrect: {params.code_path}"

    encoder = Encoder(params.model_path, params.BPE_path, )

    with open(params.code_path, "r", errors="ignore", encoding="utf8") as f:
        input_ = f.read()

    with torch.no_grad():
        output = encoder.embed(input_)
    print(output)
