import numpy as np
import pickle

def evaluate(preds, gts_path):
    """
    Given the list of all model outputs (logits), and the path to the ground 
    truth (val.pkl), calculate the percentage of correctly classified segments
    (model accuracy).
    Args:
        preds (List[torch.Tensor]): The model ouputs (logits). This is a 
            list of all the tensors produced by the model for all samples in
            val.pkl. It should be a list of length 3750 (size of val). All 
            tensors in the list should be of size 10 (number of classes).
        gts_path (str): The path to val.pkl
    Returns:
        raw_score (float): A float representing the percentage of correctly
            classified segments in val.pkl
    """
    gts = pickle.load(open(gts_path, 'rb')) # Ground truth labels, pass path to val.pkl

    raw_scores = []
    for i in range(len(preds)):
        filename, _, gt, _ = gts[i]           # Ground truth of form (filename, spectrogram, label, samples)
        logits = preds[i].cpu().numpy()       # A 10D vector that assigns probability to each class
        prediction = np.argmax(logits)        # Most confident class is the model prediction
        raw_scores.append((prediction == gt)) # Boolean: 1 if prediction is correct, 0 if incorrect

    print("ACCUARCY SCORES:")
    print("-------------------------------------------------------------")
    print()
    print('RAW: {:.2f}'.format(np.mean(raw_scores) * 100.0))
    print()
    print("-------------------------------------------------------------")

    return np.mean(raw_scores) * 100.0 # Return scores if you wish to save to a file