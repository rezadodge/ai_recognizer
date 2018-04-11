import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        lowest_bic_value = float("inf")
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = self.base_model(num_states)
            try:
                logL = hmm_model.score(self.X, self.lengths)
            except:
                continue
            logN = np.log(self.X.shape[0])
            p = num_states * (num_states - 1) + 2 * self.X.shape[1] * num_states
            bic_value = -2 * logL + p * logN

            if bic_value < lowest_bic_value:
                lowest_bic_value = bic_value
                best_model = hmm_model

        return best_model

"""
class SelectorDIC_1(ModelSelector):     ### PASSED THE TEST
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select_0(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_states = None
        highest_dic_value = float("-inf")
        for num_states in range(self.min_n_components, self.max_n_components + 1):

            likelihood_term = self.base_model(num_states).score(self.X, self.lengths)

            m = len(self.words)
            all_other_words = set(self.words.keys())
            all_other_words.discard(self.this_word)

            anti_likelihood_term = 0
            for word in all_other_words:
                model_selector = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.verbose)
                hmm_model = model_selector.base_model(num_states)
                try:
                    anti_likelihood_term += hmm_model.score(model_selector.X, model_selector.lengths)
                except:
                    m -= 1
            dic_value = likelihood_term - anti_likelihood_term / (m - 1)


            if dic_value > highest_dic_value:
                highest_dic_value = dic_value
                best_num_states = num_states
        if best_num_states is not None:
            return self.base_model(best_num_states)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_states = None
        highest_dic_value = float("-inf")
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            likelihood_term = anti_likelihood_term = 0
            try:
                likelihood_term = self.base_model(num_states).score(self.X, self.lengths)
            except:
                continue

            all_other_words = set(self.words.keys())
            m = len(all_other_words)
            all_other_words.discard(self.this_word)

            for word in all_other_words:
                model_selector = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.verbose)
                hmm_model = model_selector.base_model(num_states)
                try:
                    anti_likelihood_term += hmm_model.score(model_selector.X, model_selector.lengths)
                except:
                    m -= 1
            dic_value = likelihood_term - anti_likelihood_term / (m - 1)


            if dic_value > highest_dic_value:
                highest_dic_value = dic_value
                best_num_states = num_states
        if best_num_states is not None:
            return self.base_model(best_num_states)


class SelectorDIC_2(ModelSelector):       #### PASSED THE TEST
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_states = None
        highest_dic_value = float("-inf")
        for num_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                likelihood_term = self.base_model(num_states).score(self.X, self.lengths)
            except:
                continue

            other_words = set(self.words.keys())
            other_words.discard(self.this_word)

            model_selector = lambda word: ModelSelector(self.words, self.hwords, word,
                                                        self.n_constant, self.min_n_components,
                                                        self.max_n_components, self.verbose)
            try:
                ave_anti_likelihood_term = \
                    np.average([model_selector(word).base_model(num_states).score(
                        model_selector(word).X, model_selector(word).lengths) for word in other_words])
            except:
                m = len(self.words)
                anti_likelihood_term = 0
                for word in other_words:
                    model_selector = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.verbose)
                    hmm_model = model_selector.base_model(num_states)
                    try:
                        anti_likelihood_term += hmm_model.score(model_selector.X, model_selector.lengths)
                    except:
                        m -= 1
                ave_anti_likelihood_term = anti_likelihood_term / (m - 1)
            dic_value = likelihood_term - ave_anti_likelihood_term

            if dic_value > highest_dic_value:
                highest_dic_value = dic_value
                best_num_states = num_states
        if best_num_states is not None:
            return self.base_model(best_num_states)
"""


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    model_dict = {}
    dic_value = {}

    def generate_models_and_dictionary(cls, instance):
        for num_states in range(instance.min_n_components, instance.max_n_components + 1):
            '''
            ## COMPRESSED FORM
            SelectorDIC.model_dict[num_states] = {word:ModelSelector(instance.words, instance.hwords, word, instance.n_constant,
                                               instance.min_n_components, instance.max_n_components, instance.verbose).base_model(num_states)
                                          for word in all_words}
            '''
            SelectorDIC.model_dict[num_states] = {}
            SelectorDIC.dic_value[num_states] = {}
            other_words = set(instance.words.keys())
            for word in instance.words.keys():
                model_selector = ModelSelector(instance.words, instance.hwords, word, instance.n_constant,
                                               instance.min_n_components, instance.max_n_components, instance.verbose)
                hmm_model = model_selector.base_model(num_states)
                try:
                    likelihood_term = hmm_model.score(model_selector.X, model_selector.lengths)
                except:
                    SelectorDIC.dic_value[num_states][word] = float("-inf")
                    continue
                SelectorDIC.model_dict[num_states][word] = hmm_model
                other_words.discard(word)
                anti_likelihood_term = 0
                item_count = 0
                for item in other_words:
                    try:
                        anti_likelihood_term += SelectorDIC.model_dict[num_states][item].score(model_selector.X, model_selector.lengths)
                        item_count += 1
                    except:
                        pass
                if item_count == 0:
                    ave_anti_likelihood_term = 0
                else:
                    ave_anti_likelihood_term = anti_likelihood_term / item_count
                SelectorDIC.dic_value[num_states][word] = likelihood_term - ave_anti_likelihood_term
                other_words.add(word)

    def clear_records(self):
        SelectorDIC.dic_value = {}
        SelectorDIC.model_dict = {}

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(SelectorDIC.model_dict) == 0:
            self.generate_models_and_dictionary(self)

        best_num_states = None
        highest_dic_value = float("-inf")
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            dic_value = SelectorDIC.dic_value[num_states][self.this_word]
            if dic_value > highest_dic_value:
                highest_dic_value = dic_value
                best_num_states = num_states
        if best_num_states is not None:
            return self.base_model(best_num_states)


class SelectorCV(ModelSelector):
    '''
    select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float("-inf")
        max_n_split = 4
        n_splits_CV = min(max_n_split, len(self.sequences))
        kf = KFold(n_splits = n_splits_CV)
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)
            logL = 0
            count = n_splits_CV
            for train_index, test_index in kf.split(self.sequences):
                train_X, train_lengths = combine_sequences(train_index, self.sequences)
                test_X, test_lengths = combine_sequences(test_index, self.sequences)
                try:
                    logL += hmm_model.fit(train_X, train_lengths).score(test_X, test_lengths)
                except:
                    count -= 1

            if count != 0 and logL / count > best_score:
                best_model = hmm_model

        return best_model
