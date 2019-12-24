import random
import time

import numpy as np


from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from utils import nearestPD
import tensorflow as tf

class HRSWE(object):

    def __init__(self, *hyps, **config):

        beta1, beta2 = hyps

        self.beta1 = beta1
        self.beta2 = beta2

        self.config = config
        self.adj_pos = config['adj_pos']
        self.adj_neg = config['adj_neg']

    def specialize_emb(self,emb):
        """
        Dynamic Adjusted Word Embedding

        :param float beta:
        :param list words:
        :param nxd ndarray emb:
        :param dict thesauri: keys, [name, word_sim_pairs(if you use word sim data to create graph)]
        :param dict config:  keys, [sim_mat_type(pd/n, n means normal), eig_vec_option(ld, largest eig value),
        emb_type]

        :return: nxd ndarray new word embedding
        """

        d, n = emb.shape
        W = emb.T @ emb

        W_prime = W + self.beta1 * self.adj_pos.multiply(np.max(W) - W) + self.beta2 * self.adj_neg.multiply(W - np.min(W))

        if self.config['sim_mat_type'] == 'pd':
            W_hat = nearestPD(W_prime)
        elif self.config['sim_mat_type'] == 'n':
            W_hat = W_prime

        if self.config['eig_vec_option'] == 'ld':
            # choose d largest eigenvectors
            # ref: https://stackoverflow.com/a/12168664
            # turbo: use divide and conquer algorithm or not
            lamb_s, Q_s = linalg.eigh(W_hat, eigvals=(n - d, n - 1))

        if self.config['emb_type'] == 0:  # together use with pd
            new_emb = Q_s @ np.diag(lamb_s ** (1 / 2))
        elif self.config['emb_type'] == 1:  # together use with n
            row_norm = np.linalg.norm(Q_s, axis=1)[:, np.newaxis]
            new_emb = Q_s / row_norm
        # elif config['emb_type'] == 2: # together use with n
        #     assert np.all(lamb_s) >= 0
        #     new_emb = Q_s @ np.diag(lamb_s ** (1 / 2))
        return new_emb

class AR(object):

    def __init__(self, *hyps, **config):
        """
        To initialise the class, we need to supply the config file, which contains the location of
        the pretrained (distributional) word vectors, the location of (potentially more than one)
        collections of linguistic constraints (one pair per line), as well as the
        hyperparameters of the Attract-Repel procedure (as detailed in the TACL paper).
        """
        syn_mar, ant_mar, batch_size, epoch_num, l2_reg = hyps
        self.attract_margin_value = syn_mar
        self.repel_margin_value = ant_mar
        self.regularisation_constant_value = l2_reg
        self.batch_size = batch_size
        self.max_iter = epoch_num

        self.syn_pairs = config['syn_pairs']
        self.ant_pairs = config['ant_pairs']

        print("\nExperiment hyperparameters (attract_margin, repel_margin, batch_size, l2_reg_constant, max_iter):", \
              self.attract_margin_value, self.repel_margin_value, self.batch_size, self.regularisation_constant_value,
              self.max_iter)

    def initialise_model(self, numpy_embedding):
        """
        Initialises the TensorFlow Attract-Repel model.
        """
        self.attract_examples = tf.placeholder(tf.int32, [None, 2])  # each element is the position of word vector.
        self.repel_examples = tf.placeholder(tf.int32, [None, 2])  # each element is again the position of word vector.

        self.negative_examples_attract = tf.placeholder(tf.int32, [None, 2])
        self.negative_examples_repel = tf.placeholder(tf.int32, [None, 2])

        self.attract_margin = tf.placeholder("float")
        self.repel_margin = tf.placeholder("float")
        self.regularisation_constant = tf.placeholder("float")

        # Initial (distributional) vectors. Needed for L2 regularisation.
        self.W_init = tf.constant(numpy_embedding, name="W_init") # nxd

        # Variable storing the updated word vectors.
        self.W_dynamic = tf.Variable(numpy_embedding, name="W_dynamic") # nxd

        # Attract Cost Function:

        # placeholders for example pairs...
        # IMPORTANT: the embeddings are all normalized
        attract_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 0]),
                                                   1)
        attract_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 1]),
                                                    1)

        # and their respective negative examples:
        negative_examples_attract_left = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 0]), 1)
        negative_examples_attract_right = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 1]), 1)

        # dot product between the example pairs.
        attract_similarity_between_examples = tf.reduce_sum(tf.multiply(attract_examples_left, attract_examples_right),
                                                            1)

        # dot product of each word in the example with its negative example.
        attract_similarity_to_negatives_left = tf.reduce_sum(
            tf.multiply(attract_examples_left, negative_examples_attract_left), 1)
        attract_similarity_to_negatives_right = tf.reduce_sum(
            tf.multiply(attract_examples_right, negative_examples_attract_right), 1)

        # and the final Attract Cost Function (sans regularisation):
        self.attract_cost = tf.nn.relu(
            self.attract_margin + attract_similarity_to_negatives_left - attract_similarity_between_examples) + \
                            tf.nn.relu(
                                self.attract_margin + attract_similarity_to_negatives_right - attract_similarity_between_examples)

        # Repel Cost Function:

        # placeholders for example pairs...
        repel_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 0]),
                                                 1)  # becomes batch_size X vector_dimension
        repel_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 1]), 1)

        # and their respective negative examples:
        negative_examples_repel_left = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 0]), 1)
        negative_examples_repel_right = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 1]), 1)

        # dot product between the example pairs.
        repel_similarity_between_examples = tf.reduce_sum(tf.multiply(repel_examples_left, repel_examples_right),
                                                          1)  # becomes batch_size again, might need tf.squeeze

        # dot product of each word in the example with its negative example.
        repel_similarity_to_negatives_left = tf.reduce_sum(
            tf.multiply(repel_examples_left, negative_examples_repel_left), 1)
        repel_similarity_to_negatives_right = tf.reduce_sum(
            tf.multiply(repel_examples_right, negative_examples_repel_right), 1)

        # and the final Repel Cost Function (sans regularisation):
        self.repel_cost = tf.nn.relu(
            self.repel_margin - repel_similarity_to_negatives_left + repel_similarity_between_examples) + \
                          tf.nn.relu(
                              self.repel_margin - repel_similarity_to_negatives_right + repel_similarity_between_examples)

        # The Regularisation Cost (separate for the two terms, depending on which one is called):

        # load the original distributional vectors for the example pairs:
        original_attract_examples_left = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 0])
        original_attract_examples_right = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 1])

        original_repel_examples_left = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 0])
        original_repel_examples_right = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 1])

        # and then define the respective regularisation costs:
        regularisation_cost_attract = self.regularisation_constant * (
                    tf.nn.l2_loss(original_attract_examples_left - attract_examples_left) + tf.nn.l2_loss(
                original_attract_examples_right - attract_examples_right))
        self.attract_cost += regularisation_cost_attract

        regularisation_cost_repel = self.regularisation_constant * (
                    tf.nn.l2_loss(original_repel_examples_left - repel_examples_left) + tf.nn.l2_loss(
                original_repel_examples_right - repel_examples_right))
        self.repel_cost += regularisation_cost_repel

        # Finally, we define the training step functions for both steps.

        tvars = tf.trainable_variables()
        attract_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.attract_cost, tvars)]
        # debug: check if gradients are only calculated for the words in lexical constraints
        # self.un_clipped_attract_grads = tf.gradients(self.attract_cost, tvars)
        repel_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.repel_cost, tvars)]

        attract_optimiser = tf.train.AdagradOptimizer(0.05)
        repel_optimiser = tf.train.AdagradOptimizer(0.05)

        self.attract_cost_step = attract_optimiser.apply_gradients(list(zip(attract_grads, tvars)))
        self.repel_cost_step = repel_optimiser.apply_gradients(list(zip(repel_grads, tvars)))

        # return the handles for loading vectors from the TensorFlow embeddings:
        return attract_examples_left, attract_examples_right, repel_examples_left, repel_examples_right

    def extract_negative_examples(self, list_minibatch, attract_batch=True):
        """
        For each example in the minibatch, this method returns the closest vector which is not
        in each words example pair.
        """

        list_of_representations = []
        list_of_indices = []

        representations = self.sess.run([self.embedding_attract_left, self.embedding_attract_right],
                                        feed_dict={self.attract_examples: list_minibatch})

        for idx, (example_left, example_right) in enumerate(list_minibatch):
            list_of_representations.append(representations[0][idx])
            list_of_representations.append(representations[1][idx])

            list_of_indices.append(example_left)
            list_of_indices.append(example_right)

        condensed_distance_list = pdist(list_of_representations, 'cosine')
        square_distance_list = squareform(condensed_distance_list)

        if attract_batch:
            default_value = 2.0  # value to set for given attract/repel pair, so that it can not be found as closest or furthest away.
        else:
            default_value = 0.0  # for antonyms, we want the opposite value from the synonym one. Cosine Distance is [0,2].

        for i in range(len(square_distance_list)):

            square_distance_list[i, i] = default_value

            if i % 2 == 0:
                square_distance_list[i, i + 1] = default_value
            else:
                square_distance_list[i, i - 1] = default_value

        if attract_batch:
            negative_example_indices = np.argmin(square_distance_list,
                                                    axis=1)  # for each of the 100 elements, finds the index which has the minimal cosine distance (i.e. most similar).
        else:
            negative_example_indices = np.argmax(square_distance_list,
                                                    axis=1)  # for antonyms, find the least similar one.

        negative_examples = []

        for idx in range(len(list_minibatch)):
            negative_example_left = list_of_indices[negative_example_indices[2 * idx]]
            negative_example_right = list_of_indices[negative_example_indices[2 * idx + 1]]

            negative_examples.append((negative_example_left, negative_example_right))

        # todo: try to remove mix_sampling
        negative_examples = self.mix_sampling(list_minibatch, negative_examples)

        return negative_examples

    def mix_sampling(self,list_of_examples, negative_examples):
        """
        Converts half of the negative examples to random words from the batch (that are not in the given example pair).
        """
        mixed_negative_examples = []
        batch_size = len(list_of_examples)

        for idx, (left_idx, right_idx) in enumerate(negative_examples):

            new_left = left_idx
            new_right = right_idx

            if random.random() >= 0.5:
                new_left = list_of_examples[self.random_different_from(batch_size, idx)][random.randint(0, 1)]

            if random.random() >= 0.5:
                new_right = list_of_examples[self.random_different_from(batch_size, idx)][random.randint(0, 1)]

            mixed_negative_examples.append((new_left, new_right))

        return mixed_negative_examples

    def random_different_from(self,top_range, number_to_not_repeat):
        result = random.randint(0, top_range - 1)
        while result == number_to_not_repeat:
            result = random.randint(0, top_range - 1)

        return result

    def obtain_sp_vec(self):
        """
        Extracts the current word vectors from TensorFlow embeddings and (if print_simlex=True) prints their SimLex scores.
        """

        return self.sess.run(self.W_dynamic)

    def sepcialize_emb(self,emb):

        """
        This method repeatedly applies optimisation steps to fit the word vectors to the provided linguistic constraints.
        """

        # load the handles so that we can load current state of vectors from the Tensorflow embedding.
        tf.reset_default_graph()
        embedding_handles = self.initialise_model(emb)

        self.embedding_attract_left = embedding_handles[0]
        self.embedding_attract_right = embedding_handles[1]
        self.embedding_repel_left = embedding_handles[2]
        self.embedding_repel_right = embedding_handles[3]

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

        # specialize embedding
        current_iteration = 0

        # todo: do post-processing outside and remove the following
        # # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
        # for antonym_pair in self.antonyms:
        #     if antonym_pair in self.synonyms:
        #         self.synonyms.remove(antonym_pair)

        # TODO: if a pair (w_a,w_b) is in synonyms, (w_b,w_a) can also in synonyms.
        # figure out how many pairs have this problem

        # self.synonyms = list(self.synonyms)
        # self.antonyms = list(self.antonyms)

        syn_count = len(self.syn_pairs)
        ant_count = len(self.ant_pairs)

        print("\nAntonym pairs:", len(self.ant_pairs), "Synonym pairs:", len(self.syn_pairs))

        syn_batches = int(syn_count / self.batch_size)
        ant_batches = int(ant_count / self.batch_size)

        batches_per_epoch = syn_batches + ant_batches

        print("\nRunning the optimisation procedure for", self.max_iter, "iterations...")
        start = time.time()
        last_time = time.time()


        while current_iteration < self.max_iter:

            # how many attract/repel batches we've done in this epoch so far.
            antonym_batch_counter = 0
            synonym_batch_counter = 0

            order_of_synonyms = list(range(0, syn_count))
            order_of_antonyms = list(range(0, ant_count))
            # IMPORTANT: before training the syn/ant constraints will be shuffled
            random.shuffle(order_of_synonyms)
            random.shuffle(order_of_antonyms)

            # list of 0 where we run synonym batch, 1 where we run antonym batch
            # IMPORTANT: training steps of syn and ant mini batches will be shuffled
            list_of_batch_types = [0] * batches_per_epoch
            list_of_batch_types[syn_batches:] = [1] * ant_batches  # all antonym batches to 1
            random.shuffle(list_of_batch_types)

            if current_iteration == 0:
                print("\nStarting epoch:", current_iteration + 1, "\n")
            else:
                print("\nStarting epoch:", current_iteration + 1, "Last epoch took:",
                      round(time.time() - last_time, 1),
                      "seconds. \n")
                last_time = time.time()

            for batch_index in range(0, batches_per_epoch):

                syn_or_ant_batch = list_of_batch_types[batch_index]

                if syn_or_ant_batch == 0:
                    # do one synonymy batch:

                    synonymy_examples = [self.syn_pairs[order_of_synonyms[x]] for x in
                                         range(synonym_batch_counter * self.batch_size,
                                               (synonym_batch_counter + 1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(synonymy_examples, attract_batch=True)

                    self.sess.run([self.attract_cost_step], feed_dict={self.attract_examples: synonymy_examples,
                                                                       self.negative_examples_attract: current_negatives, \
                                                                       self.attract_margin: self.attract_margin_value,
                                                                       self.regularisation_constant: self.regularisation_constant_value})
                    synonym_batch_counter += 1

                else:

                    antonymy_examples = [self.ant_pairs[order_of_antonyms[x]] for x in
                                         range(antonym_batch_counter * self.batch_size,
                                               (antonym_batch_counter + 1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(antonymy_examples, attract_batch=False)

                    self.sess.run([self.repel_cost_step], feed_dict={self.repel_examples: antonymy_examples,
                                                                     self.negative_examples_repel: current_negatives, \
                                                                     self.repel_margin: self.repel_margin_value,
                                                                     self.regularisation_constant: self.regularisation_constant_value})

                    antonym_batch_counter += 1

            current_iteration += 1

        print("Training took ", round(time.time() - start, 1), "seconds.")

        return self.obtain_sp_vec()