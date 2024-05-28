import logging
from collections import defaultdict


class Scores:
    def __init__(self):
        self.u = 0
        self.prediction, self.ground_truth = defaultdict(set), defaultdict(set)
    def update(self, preds, truths):
        self.prediction[self.u] = set(preds)
        self.ground_truth[self.u] = set(truths)
        self.u += 1
    @property
    def evaluate(self):
        """
        Evaluation matrix.
        :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
        :param ground_truth: a dictionary of labels
        :return:
        """
        print("prediction:%d, ground:%d"%(len(self.prediction),len(self.ground_truth)))
        assert len(self.prediction) == len(self.ground_truth)
        count = len(self.prediction)
        # print 'Test', count, 'mentions'
        info = {
            'same': 0, 'macro_precision': 0.0, 'macro_recall': 0.0,
            'micro_n': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0
        }
        for i in self.ground_truth:
            p, g = self.prediction[i], self.ground_truth[i]
            if p == g:
                info['same'] += 1
            same_count = len(p & g)
            info['macro_precision'] += float(same_count) / float(len(p))
            info['macro_recall'] += float(same_count) / float(len(g))
            info['micro_n'] += same_count
            info['micro_precision'] += len(p)
            info['micro_recall'] += len(g)
        info['accuracy'] = float(info['same']) / float(count)
        info['macro_precision'] /= count
        info['macro_recall'] /= count
        info['macro_f1'] = 2 * info['macro_precision'] * info['macro_recall'] / \
            (info['macro_precision'] + info['macro_recall'] + 1e-8)
        info['micro_precision'] = info['micro_n'] / info['micro_precision']
        info['micro_recall'] = info['micro_n'] / info['micro_recall']
        info['micro_f1'] = 2 * info['micro_precision'] * info['micro_recall'] / \
            (info['micro_precision'] + info['micro_recall'] + 1e-8)
        return info 

if __name__ == '__main__':
    scores = Scores()
    scores.update([1, 2], [1, 2, 3])
    logging.basicConfig(filename='./example.log', level=logging.INFO)
    print(scores.evaluate)
    logging.info(scores.evaluate)
    