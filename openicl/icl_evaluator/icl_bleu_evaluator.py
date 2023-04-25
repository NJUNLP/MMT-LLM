'''BLEU evaluator'''
from openicl.icl_evaluator import IclBaseEvaluator
from openicl.utils.metrics import get_ngrams
from typing import List, Dict
import evaluate
import math
import collections
import sys
import os

class IclBleuEvaluator(IclBaseEvaluator):
    def __init__(self, references: List, max_order=4, smooth=False) -> None:
        super().__init__(references)
        self.max_order = max_order
        self.smooth = smooth
    
    
    def score(self, predictions, references=None, src_lang=None, tgt_lang=None):
        if references == None:
            references = self.references
        assert len(predictions) == len(references)
        pred_dict = {}
        ref_dict = {}
        for idx in range(len(predictions)):
            pred_dict[idx] = predictions[idx].split()
            ref_dict[idx] = references[idx].split()
        return self.calculate_bleu(ref_dict, pred_dict, self.max_order, self.smooth, src_lang, tgt_lang)
            
            
    def calculate_bleu(self, references: Dict[str, List[List[str]]],
                    predictions: Dict[str, List[str]],
                    max_order=4,
                    smooth=False,
                    src_lang=None,
                    tgt_lang=None) -> float:

        reference_corpus = []
        prediction_corpus = []

        for instance_id, reference_sents in references.items():
            try:
                prediction_sent = predictions[instance_id]
            except KeyError:
                print("Missing prediction for instance '%s'.", instance_id)
                sys.exit(-1)

            del predictions[instance_id]

            prediction_corpus.append(prediction_sent)
            reference_corpus.append(reference_sents)

        if len(predictions) > 0:
            raise LookupError("Found %d extra predictions, for example: %s", len(predictions),
                        ", ".join(list(predictions.keys())[:3]))

        # add by zhuwh
        _reference_corpus = [' '.join(tok_list)+'\n' for tok_list in reference_corpus]
        _prediction_corpus = [' '.join(tok_list)+'\n' for tok_list in prediction_corpus]

        # with open('results/{}_{}_hyp.txt'.format(src_lang, tgt_lang), 'w') as f:
        #     f.writelines(_prediction_corpus)

        # with open('results/{}_{}_ref.txt'.format(src_lang, tgt_lang), 'w') as f:
        #     f.writelines(_reference_corpus)

        # score = self._compute_bleu(reference_corpus, prediction_corpus,
        #                     max_order=max_order, smooth=smooth)

        ## add by liuhy
        import sacrebleu
        score = sacrebleu.corpus_bleu(_prediction_corpus, [_reference_corpus], tokenize="spm")

        return score.score
    
    
    def _compute_bleu(self, reference_corpus, translation_corpus, max_order=4, smooth=False):
        """Computes BLEU score of translated segments against one or more references.
        Args:
            reference_corpus: list of lists of references for each translation. Each
                reference should be tokenized into a list of tokens.
            translation_corpus: list of translations to score. Each translation
                should be tokenized into a list of tokens.
            max_order: Maximum n-gram order to use when computing BLEU score.
            smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
            3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
                precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        for (references, translation) in zip(reference_corpus, translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= get_ngrams(reference, max_order)
            translation_ngram_counts = get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]
            for order in range(1, max_order+1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                    possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        # return (bleu, precisions, bp, ratio, translation_length, reference_length)
        return {
            'BLEU': bleu,
            'precisions': precisions,
            'bp': bp,
            'ratio': ratio,
            'translation_length': translation_length,
            'reference_length': reference_length
        }