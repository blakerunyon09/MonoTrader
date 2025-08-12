import numpy as np
class RandomTreeLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.data_x = None
        self.data_y = None
        self.tree = None

    def g_impurity(self, data_y):
        _, c = np.unique(data_y, return_counts=True)
        return 1 - np.sum((c / c.sum()) ** 2)

    def build_tree(self, data_x, data_y, mode="gini"):
        # Base Case -> Leaf Node
        if data_x.shape[0] <= self.leaf_size:
            v, c = np.unique(data_y, return_counts=True)
            return np.array([[-1, v[np.argmax(c)], np.nan, np.nan]])
        
        # Base Case -> All y values are the same
        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])

        if mode == "gini":
            split_index = np.random.randint(data_x.shape[1])
            feature = data_x[:, split_index]
            feature_values = np.unique(feature)
            possible_split_points = (feature_values[:-1] + feature_values[1:]) / 2

            best_split = None
            best_gini = float('-inf')
            parent_gini = self.g_impurity(data_y)

            for split_point in possible_split_points:
                left_mask = feature <= split_point
                right_mask = feature > split_point

                if not left_mask.any() or not right_mask.any():
                    continue

                left_count = left_mask.sum()
                right_count = right_mask.sum()

                left_gini = self.g_impurity(data_y[left_mask])
                right_gini = self.g_impurity(data_y[right_mask])

                gain = parent_gini - (left_count / data_x.shape[0]) * left_gini - (right_count / data_x.shape[0]) * right_gini

                if best_gini < gain:
                    best_gini = gain
                    best_split = split_point

            if best_split is None:
                v, c = np.unique(data_y, return_counts=True)
                return np.array([[-1, v[np.argmax(c)], np.nan, np.nan]])
            
            left_mask = feature <= best_split
            right_mask = feature > best_split

            left_tree = self.build_tree(data_x[left_mask], data_y[left_mask], mode=mode)
            right_tree = self.build_tree(data_x[right_mask], data_y[right_mask], mode=mode)

            root = np.array([split_index, best_split, 1, left_tree.shape[0] + 1])

            return np.vstack((root, left_tree, right_tree))

        elif mode == "median":
            # Find highest rxy
            split_index = np.random.randint(data_x.shape[1])
            split_val = np.median(data_x[:, split_index])

            left_mask = data_x[:, split_index] <= split_val
            right_mask = data_x[:, split_index] > split_val

            # Check that both masks have data
            if np.all(left_mask) or np.all(right_mask):
                values, counts = np.unique(data_y, return_counts=True)
                mode_result = values[np.argmax(counts)]
                return np.array([[-1, mode_result, np.nan, np.nan]])

            left_tree = self.build_tree(data_x[left_mask], data_y[left_mask], mode=mode)
            right_tree = self.build_tree(data_x[right_mask], data_y[right_mask], mode=mode)

            root = np.array([split_index, split_val, 1, left_tree.shape[0] + 1])

            return np.vstack((root, left_tree, right_tree))


    def add_evidence(self, data_x, data_y, mode="gini"):
        self.data_x = data_x
        self.data_y = data_y
        self.tree = self.build_tree(self.data_x, self.data_y, mode=mode)

    def query(self, points):
        predictions = np.ndarray(points.shape[0])

        for i in range(points.shape[0]):
            point = points[i]
            cursor = 0

            while True:
                current_node = self.tree[cursor] 
                split_index = int(current_node[0])

                if split_index == -1:
                    predictions[i] = current_node[1]
                    break
                
                split_value = current_node[1]
                left_offset = int(current_node[2])
                right_offset = int(current_node[3])

                if point[split_index] <= split_value:
                    cursor += left_offset
                else:
                    cursor += right_offset

        return predictions