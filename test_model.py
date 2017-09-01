import unittest
import tensorflow as tf
from model import *


class TestWWHSInference(unittest.TestCase):

    def _print_success_message(self):
        print('Tests Passed')

    def test_get_tensors(self):
        test_graph = tf.Graph()
        with test_graph.as_default():
            test_input = tf.placeholder(tf.int32, name='input')
            test_initial_state = tf.placeholder(tf.int32, name='initial_state')
            test_final_state = tf.placeholder(tf.int32, name='final_state')
            test_probs = tf.placeholder(tf.float32, name='probs')

        input_text, initial_state, final_state, probs = get_tensors(test_graph)

        # Check correct tensor
        assert input_text == test_input, \
            'Test input is wrong tensor'
        assert initial_state == test_initial_state, \
            'Initial state is wrong tensor'
        assert final_state == test_final_state, \
            'Final state is wrong tensor'
        assert probs == test_probs, \
            'Probabilities is wrong tensor'

        self._print_success_message()

    def test_pick_word(self):
        with tf.Graph().as_default():
            test_probabilities = np.array([0.1, 0.8, 0.05, 0.05])
            test_int_to_vocab = {word_i: word for word_i, word in enumerate(['this', 'is', 'a', 'test'])}

            pred_word = pick_word(test_probabilities, test_int_to_vocab)

            # Check type
            assert isinstance(pred_word, str), \
                'Predicted word is wrong type. Found {} type.'.format(type(pred_word))

            # Check word is from vocab
            assert pred_word in test_int_to_vocab.values(), \
                'Predicted word not found in int_to_vocab.'

        self._print_success_message()


if __name__ == '__main__':
    unittest.main()
