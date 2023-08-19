"""
Genetic Algorithm Word Swap
====================================
"""
from abc import ABC, abstractmethod

import numpy as np
import torch

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared.validators import transformation_consists_of_word_swaps

import itertools
import re
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from textattack.shared import utils
from textattack.transformations.word_swaps import WordSwap

class WordSwapMaskedLM2(WordSwap):

    def __init__(
        self,
        method="bae",
        masked_language_model="bert-base-uncased",
        tokenizer=None,
        max_length=512,
        window_size=float("inf"),
        max_candidates=50,
        min_confidence=5e-4,
        batch_size=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.max_length = max_length
        self.window_size = window_size
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size

        if isinstance(masked_language_model, str):
            print("if")
            print(masked_language_model)
            self._language_model = AutoModelForMaskedLM.from_pretrained(
                masked_language_model
            )
            self._lm_tokenizer = AutoTokenizer.from_pretrained(
                masked_language_model, use_fast=True
            )
        else:
            print("else")
            print(masked_language_model)
            self._language_model = masked_language_model
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self._lm_tokenizer = tokenizer
        self._language_model.to(utils.device)
        self._language_model.eval()
        self.masked_lm_name = self._language_model.__class__.__name__

    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding.to(utils.device)

    def _bae_replacement_words(self, current_text, indices_to_modify):
        """Get replacement words for the word we want to replace using BAE
        method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
        """
        masked_texts = []
#         print('current_text : ', current_text)
#         print('indices_to_modify : ', indices_to_modify)
        for index in indices_to_modify:
            masked_text = current_text.replace_word_at_index(
                index, self._lm_tokenizer.mask_token
            )
            masked_texts.append(masked_text.text)
#         print('masked_texts: ', masked_texts)
        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        replacement_words = []
        while i < len(masked_texts):
            inputs = self._encode_text(masked_texts[i : i + self.batch_size])
            ids = inputs["input_ids"].tolist()
            with torch.no_grad():
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    replacement_words.append([])
                    continue

                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                top_words = []
                for _id in ranked_indices:
                    _id = _id.item()
                    word = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    if utils.check_if_subword(
                        word,
                        self._language_model.config.model_type,
                        (masked_index == 1),
                    ):
                        word = utils.strip_BPE_artifacts(
                            word, self._language_model.config.model_type
                        )
                    if (
                        mask_token_probs[_id] >= self.min_confidence
                        and utils.is_one_word(word)
                        and not utils.check_if_punctuations(word)
                    ):
                        top_words.append(word)

                    if (
                        len(top_words) >= self.max_candidates
                        or mask_token_probs[_id] < self.min_confidence
                    ):
                        break

                replacement_words.append(top_words)

            i += self.batch_size

        return replacement_words

    def _bert_attack_replacement_words(
        self,
        current_text,
        index,
        id_preds,
        masked_lm_logits,
    ):
        """Get replacement words for the word we want to replace using BERT-
        Attack method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
            id_preds (torch.Tensor): N x K tensor of top-K ids for each token-position predicted by the masked language model.
                N is equivalent to `self.max_length`.
            masked_lm_logits (torch.Tensor): N x V tensor of the raw logits outputted by the masked language model.
                N is equivlaent to `self.max_length` and V is dictionary size of masked language model.
        """
        # We need to find which BPE tokens belong to the word we want to replace
        masked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )
        current_inputs = self._encode_text(masked_text.text)
        current_ids = current_inputs["input_ids"].tolist()[0]
        word_tokens = self._lm_tokenizer.encode(
            current_text.words[index], add_special_tokens=False
        )

        try:
            # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        # List of indices of tokens that are part of the target word
        target_ids_pos = list(
            range(masked_index, min(masked_index + len(word_tokens), self.max_length))
        )

        if not len(target_ids_pos):
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            top_preds = id_preds[target_ids_pos[0]].tolist()
            replacement_words = []
            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if utils.is_one_word(token) and not utils.check_if_subword(
                    token, self._language_model.config.model_type, index == 0
                ):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_preds = [id_preds[i] for i in target_ids_pos]
            products = itertools.product(*top_preds)
            combination_results = []
            # Original BERT-Attack implement uses cross-entropy loss
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.tensor(target_ids_pos)
            word_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)
            for bpe_tokens in products:
                for i in range(len(bpe_tokens)):
                    word_tensor[i] = bpe_tokens[i]

                logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
                loss = cross_entropy_loss(logits, word_tensor)
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()
                word = "".join(
                    self._lm_tokenizer.convert_ids_to_tokens(word_tensor)
                ).replace("##", "")
                if utils.is_one_word(word):
                    combination_results.append((word, perplexity))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]
            ]
            return top_replacements

    def _get_transformations(self, current_text, indices_to_modify):
        indices_to_modify = list(indices_to_modify)
        if self.method == "bert-attack":
            current_inputs = self._encode_text(current_text.text)
            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()

            transformed_texts = []

            for i in indices_to_modify:
                word_at_index = current_text.words[i]
                replacement_words = self._bert_attack_replacement_words(
                    current_text,
                    i,
                    id_preds=id_preds,
                    masked_lm_logits=masked_lm_logits,
                )

                for r in replacement_words:
                    r = r.strip("Ġ")
                    if r != word_at_index:
                        transformed_texts.append(
                            current_text.replace_word_at_index(i, r)
                        )

            return transformed_texts

        elif self.method == "bae":
            replacement_words = self._bae_replacement_words(
                current_text, indices_to_modify
            )
#             print('replacement_words : ', replacement_words)
#             print(replacement_words.shape)
            transformed_texts = []
#             print('for loop to choose replacement words')
            for i in range(len(replacement_words)):
#                 print('replacement_words ', replacement_words[i])
                index_to_modify = indices_to_modify[i]
                word_at_index = current_text.words[index_to_modify]
#                 print('word to replace (word_at_index) ', word_at_index)
#                 print('loop over words')
                for word in replacement_words[i]:
                    word = word.strip("Ġ")
                    if (
                        word != word_at_index
                        and re.search("[\u0600-\u06FF]", word)
                        and len(utils.words_from_text(word)) == 1
                    ):
                        transformed_texts.append(
                            current_text.replace_word_at_index(index_to_modify, word)
                        )
            return transformed_texts
        else:
            raise ValueError(f"Unrecognized value {self.method} for `self.method`.")

    def extra_repr_keys(self):
        return [
            "method",
            "masked_lm_name",
            "max_length",
            "max_candidates",
            "min_confidence",
        ]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word


class GeneticAlgorithm(PopulationBasedSearch, ABC):
    """Base class for attacking a model with word substiutitions using a
    genetic algorithm.

    Args:
        pop_size (int): The population size. Defaults to 20.
        max_iters (int): The maximum number of iterations to use. Defaults to 50.
        temp (float): Temperature for softmax function used to normalize probability dist when sampling parents.
            Higher temperature increases the sensitivity to lower probability candidates.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        post_crossover_check (bool): If True, check if child produced from crossover step passes the constraints.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Applied only when `post_crossover_check` is set to `True`.
            Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
    """

    def __init__(
        self,
        pop_size=60,
        max_iters=20,
        temp=0.3,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = temp
        self.give_up_if_no_improvement = give_up_if_no_improvement
        self.post_crossover_check = post_crossover_check
        self.max_crossover_retries = max_crossover_retries

        # internal flag to indicate if search should end immediately
        self._search_over = False

    @abstractmethod
    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and, `attributes` altered appropriately for given
        `word_idx`"""
        raise NotImplementedError()

    @abstractmethod
    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        raise NotImplementedError

    def _perturb(self, pop_member, original_result, index=None):
        """Perturb `pop_member` and return it. Replaces a word at a random
        (unless `index` is specified) in `pop_member`.

        Args:
            pop_member (PopulationMember): The population member being perturbed.
            original_result (GoalFunctionResult): Result of original sample being attacked
            index (int): Index of word to perturb.
        Returns:
            Perturbed `PopulationMember`
        """
        num_words = pop_member.attacked_text.num_words
        # `word_select_prob_weights` is a list of values used for sampling one word to transform
        word_select_prob_weights = np.copy(
            self._get_word_select_prob_weights(pop_member)
        )
        non_zero_indices = np.count_nonzero(word_select_prob_weights)
        if non_zero_indices == 0:
            return pop_member
        iterations = 0
        import itertools
        import re
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from textattack.shared import utils
        from textattack.transformations.word_swaps import WordSwap
        transformation_inatance = WordSwapMaskedLM2(
                    method="bae",
                    masked_language_model='UBC-NLP/MARBERT',
                    max_candidates=8,
                    min_confidence=5e-4,
                    batch_size=16,
                    max_length=128
                    )    
        from textattack.shared import AttackedText
        while iterations < non_zero_indices:
            if index:
                idx = index
            else:
                w_select_probs = word_select_prob_weights / np.sum(
                    word_select_prob_weights
                )
                idx = np.random.choice(num_words, 1, p=w_select_probs)[0]

            #transformed_texts = self.get_transformations(
            #    pop_member.attacked_text,
            #    original_text=original_result.attacked_text,
            #    indices_to_modify=[idx],
            #)
            transformed_texts = transformation_inatance._get_transformations(pop_member.attacked_text, [idx])

            if not len(transformed_texts):
                iterations += 1
                continue

            new_results, self._search_over = self.get_goal_results(transformed_texts)

            diff_scores = (
                torch.Tensor([r.score for r in new_results]) - pop_member.result.score
            )
            if len(diff_scores) and diff_scores.max() > 0:
                idx_with_max_score = diff_scores.argmax()
                pop_member = self._modify_population_member(
                    pop_member,
                    transformed_texts[idx_with_max_score],
                    new_results[idx_with_max_score],
                    idx,
                )
                return pop_member

            word_select_prob_weights[idx] = 0
            iterations += 1

            if self._search_over:
                break

        return pop_member

    @abstractmethod
    def _crossover_operation(self, pop_member1, pop_member2):
        """Actual operation that takes `pop_member1` text and `pop_member2`
        text and mixes the two to generate crossover between `pop_member1` and
        `pop_member2`.

        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        raise NotImplementedError()

    def _post_crossover_check(
        self, new_text, parent_text1, parent_text2, original_text
    ):
        """Check if `new_text` that has been produced by performing crossover
        between `parent_text1` and `parent_text2` aligns with the constraints.

        Args:
            new_text (AttackedText): Text produced by crossover operation
            parent_text1 (AttackedText): Parent text of `new_text`
            parent_text2 (AttackedText): Second parent text of `new_text`
            original_text (AttackedText): Original text
        Returns:
            `True` if `new_text` meets the constraints. If otherwise, return `False`.
        """
        if "last_transformation" in new_text.attack_attrs:
            previous_text = (
                parent_text1
                if "last_transformation" in parent_text1.attack_attrs
                else parent_text2
            )
            passed_constraints = self._check_constraints(
                new_text, previous_text, original_text=original_text
            )
            return passed_constraints
        else:
            # `new_text` has not been actually transformed, so return True
            return True

    def _crossover(self, pop_member1, pop_member2, original_text):
        """Generates a crossover between pop_member1 and pop_member2.

        If the child fails to satisfy the constraints, we re-try crossover for a fix number of times,
        before taking one of the parents at random as the resulting child.
        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
            original_text (AttackedText): Original text
        Returns:
            A population member containing the crossover.
        """
        x1_text = pop_member1.attacked_text
        x2_text = pop_member2.attacked_text

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_crossover_retries + 1:
            new_text, attributes = self._crossover_operation(pop_member1, pop_member2)

            replaced_indices = new_text.attack_attrs["newly_modified_indices"]
            new_text.attack_attrs["modified_indices"] = (
                x1_text.attack_attrs["modified_indices"] - replaced_indices
            ) | (x2_text.attack_attrs["modified_indices"] & replaced_indices)

            if "last_transformation" in x1_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs[
                    "last_transformation"
                ]
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs[
                    "last_transformation"
                ]

            if self.post_crossover_check:
                passed_constraints = self._post_crossover_check(
                    new_text, x1_text, x2_text, original_text
                )

            if not self.post_crossover_check or passed_constraints:
                break

            num_tries += 1

        if self.post_crossover_check and not passed_constraints:
            # If we cannot find a child that passes the constraints,
            # we just randomly pick one of the parents to be the child for the next iteration.
            pop_mem = pop_member1 if np.random.uniform() < 0.5 else pop_member2
            return pop_mem
        else:
            new_results, self._search_over = self.get_goal_results([new_text])
            return PopulationMember(
                new_text, result=new_results[0], attributes=attributes
            )

    @abstractmethod
    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        raise NotImplementedError()

    def perform_search(self, initial_result):
        print(initial_result.attacked_text)
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        pop_size = len(population)
        current_score = initial_result.score

        for i in range(self.max_iters):
            population = sorted(population, key=lambda x: x.result.score, reverse=True)

            if (
                self._search_over
                or population[0].result.goal_status
                == GoalFunctionResultStatus.SUCCEEDED
            ):
                break

            if population[0].result.score > current_score:
                current_score = population[0].result.score
                print("The print in the beginning")
                print(population[0].attacked_text)
            elif self.give_up_if_no_improvement:
                print("ThIS IS THE ELIF")
                break

            pop_scores = torch.Tensor([pm.result.score for pm in population])
            logits = ((-pop_scores) / self.temp).exp()
            select_probs = (logits / logits.sum()).cpu().numpy()

            parent1_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)

            children = []
            for idx in range(pop_size - 1):
                child = self._crossover(
                    population[parent1_idx[idx]],
                    population[parent2_idx[idx]],
                    initial_result.attacked_text,
                )
                if self._search_over:
                    break

                child = self._perturb(child, initial_result)
                print(child.attacked_text)
                if child.result.score > current_score:
                    print("score is high")
                    return child.result
                else:
                    print("Score is still low")        
                children.append(child)

                # We need two `search_over` checks b/c value might change both in
                # `crossover` method and `perturb` method.
                if self._search_over:
                    break

            population = [population[0]] + children

        return population[0].result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return [
            "pop_size",
            "max_iters",
            "temp",
            "give_up_if_no_improvement",
            "post_crossover_check",
            "max_crossover_retries",
        ]
