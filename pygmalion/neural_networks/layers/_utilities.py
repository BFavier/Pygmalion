import torch
from typing import List, Tuple, Optional
from copy import deepcopy


def beam_search(n_beams: int,
                sequences: List[List[int]],
                histories: List[Tuple[dict]],
                sum_likelyhoods: List[float],
                predicted_likelyhoods: List[Optional[torch.Tensor]]):
    """
    Perform one step of the beam search algorithm.
    The input lists are modified inplace, excepted for 'predicted_likelyhoods'.

    Parameters
    ----------
    n_beams : int
        number of beams
    sequences : list of list of int
        for each beam, the sequence of already predicted tokens
    histories : list of tuple of dict
        for each beam, the prediction history
    sum_likelyhoods : list of float
        for each beam, the sumed log likelyhood over each predicted tokens
    predicted_log_likelyhood : list of torch.Tensor or None
        for each beam, the predicted log likelyhoods for each class.
        Tensor of floats of shape (n_classes,).
        If the beam's sequence already ended, None instead.
    """
    # for each beam, get the top-k (token index, log-likelyhood) of the token candidates for the beam
    # aka : [[(token_index, log_likelyhood) for k] for beam]
    topk = [torch.topk(log.reshape(-1), n_beams) if log is not None else None
            for log in predicted_likelyhoods]
    topk = [list(zip(tk.indices.detach().cpu().tolist(),
                     tk.values.detach().cpu().tolist()))
            if tk is not None else [(None, None)]
            for tk in topk]
    # get the (beam index, predicted token, mean-log-likelyhood) for the top most likely sequences
    beams = sorted([(beam, token, ll) for beam, tk in enumerate(topk) for token, ll in tk],
                   key = lambda x: (sum_likelyhoods[x[0]] + x[-1]) / (len(sequences[x[0]])+1)
                                    if x[-1] is not None else
                                    sum_likelyhoods[x[0]] / len(sequences[x[0]]),
                    reverse=True
                    )[:n_beams]
    beam_indices, token_indices, predicted_ll = zip(*beams)
    # modify inputs
    new_sequences = [list(sequences[beam]) + ([token] if token is not None else [])
                     for beam, token in zip(beam_indices, token_indices)]
    sequences.clear()
    sequences.extend(new_sequences)
    new_histories = [deepcopy(histories[beam]) for beam in beam_indices]
    histories.clear()
    histories.extend(new_histories)
    new_sum_log_likelyhoods = [sum_likelyhoods[beam] + (ll or 0) for beam, ll in zip(beam_indices, predicted_ll)]
    sum_likelyhoods.clear()
    sum_likelyhoods.extend(new_sum_log_likelyhoods)
