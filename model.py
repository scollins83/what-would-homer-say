import os
import tensorflow as tf
import numpy as np

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    initial_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    probs_tensor = loaded_graph.get_tensor_by_name("probs:0")
    return input_tensor, initial_state_tensor, final_state_tensor, probs_tensor

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-1]] = 0
    p = p / np.sum(p)
    c = np.random.choice(len(int_to_vocab), 1, p=p)[0]
    return int_to_vocab[c]