import numpy as np

class BagLearner:
  def __init__(self, learner, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False, mode="gini"):
    self.learner = learner
    self.kwargs = kwargs
    self.bags = bags
    self.boost = boost
    self.verbose = verbose
    self.trees = []
    self.mode = mode

  def add_evidence(self, data_x, data_y):
    self.data_x = data_x
    self.data_y = data_y

    for _ in range(self.bags):
        _i = np.random.randint(0, data_x.shape[0], size=data_x.shape[0])

        # Build the trees
        tree = self.learner(**self.kwargs)
        tree.add_evidence(data_x[_i], data_y[_i], self.mode)
        self.trees.append(tree)

  def query(self, points):
    result = np.zeros((points.shape[0], self.bags))

    for i, t in enumerate(self.trees):
      _r = t.query(points)
      result[:, i] = _r

    trades = np.apply_along_axis(
      lambda row: np.bincount(row.astype(int) + 1).argmax() - 1,
      axis=1,
      arr=result
    )

    return trades
