


def compute_PR(triggered, correct, ground_truth):

    precision = 1 if triggered == 0 else float(correct) / float(triggered)
    recall = 1 if ground_truth == 0 else float(correct) / float(ground_truth)

    return precision, recall