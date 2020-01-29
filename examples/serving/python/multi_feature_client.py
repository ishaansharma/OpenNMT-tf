import argparse
import os

import tensorflow as tf
import tensorflow_addons as tfa  # Register TensorFlow Addons kernels.

import pyonmttok


#length_0, length_1, length_1, tokens_0,tokens_1, tokens_2
class Translator(object):
    def __init__(self, export_dir):
        imported = tf.saved_model.load(export_dir)
        self._translate_fn = imported.signatures["serving_default"]
        self._tokenizer = pyonmttok.Tokenizer("none",joiner_annotate=True, segment_numbers=True)#https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python
        #tokens = tokenizer.tokenize(text: str)

    def translate(self, src, fea1, fea2):
        """Translates a batch of texts."""
        inputs = self._preprocess(src, fea1, fea2)
        outputs = self._translate_fn(**inputs)
        return self._postprocess(outputs)

    def _preprocess(self, src, fea1, fea2):
        
        all_tokens_src = []
        lengths_src = []
        max_length_src = 0
        for text_src in src:
            tokens_src, _ = self._tokenizer.tokenize(text_src)
            length_src = len(tokens_src)
            all_tokens_src.append(tokens_src)
            lengths_src.append(length_src)
            max_length_src = max(max_length_src, length_src)
        for tokens_src, length_src in zip(all_tokens_src, lengths_src):
            if length_src < max_length_src:
                tokens_src += [""] * (max_length_src - length_src)

        all_tokens_fea1 = []
        lengths_fea1 = []
        max_length_fea1 = 0
        for text_fea1 in fea1:
            tokens_fea1, _ = self._tokenizer.tokenize(text_fea1)
            length_fea1 = len(tokens_fea1)
            all_tokens_fea1.append(tokens_fea1)
            lengths_fea1.append(length_fea1)
            max_length_fea1 = max(max_length_fea1, length_fea1)
        for tokens_fea1, length_fea1 in zip(all_tokens_fea1, lengths_fea1):
            if length_fea1 < max_length_fea1:
                tokens_fea1 += [""] * (max_length_fea1 - length_fea1)
    
        all_tokens_fea2 = []
        lengths_fea2 = []
        max_length_fea2 = 0
        for text_fea2 in fea2:
            tokens_fea2, _ = self._tokenizer.tokenize(text_fea2)
            length_fea2 = len(tokens_fea2)
            all_tokens_fea2.append(tokens_fea2)
            lengths_fea2.append(length_fea2)
            max_length_fea2 = max(max_length_fea2, length_fea2)
        for tokens_fea2, length_fea2 in zip(all_tokens_fea2, lengths_fea2):
            if length_fea2 < max_length_fea2:
                tokens_fea2 += [""] * (max_length_fea2 - length_fea2)
        
        inputs = {
        "tokens_0": tf.constant(all_tokens_src, dtype=tf.string),
        "length_0": tf.constant(lengths_src, dtype=tf.int32),
        "tokens_1": tf.constant(all_tokens_fea1, dtype=tf.string),
        "length_1": tf.constant(lengths_fea1, dtype=tf.int32),
        "tokens_2": tf.constant(all_tokens_fea2, dtype=tf.string),
        "length_2": tf.constant(lengths_fea2, dtype=tf.int32)}
        return inputs

    def _postprocess(self, outputs):
        texts = []
        for tokens, length in zip(outputs["tokens"].numpy(), outputs["length"].numpy()):
          tokens = tokens[0][:length[0]].tolist()
          texts.append(self._tokenizer.detokenize(tokens))
        return texts


def main():
  parser = argparse.ArgumentParser(description="Translation client example")
  parser.add_argument("export_dir", help="Saved model directory")
  args = parser.parse_args()

  translator = Translator(args.export_dir)

  while True:
    source = input("source : ")
    feature_1 = input("feature_1: ")
    feature_2 = input("feature_2: ")
    output = translator.translate([source],[feature_1],[feature_2])
    print("Target: %s" % output[0])
    print("")


if __name__ == "__main__":
  main()
