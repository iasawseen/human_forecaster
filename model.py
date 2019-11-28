import numpy as np
from utils import load


class ModelWrapper:
    def __init__(self, file_path_pattern, length):
        self.model = load(file_path=file_path_pattern.format(length))
        self.length = length

    def process_chunk(self, chunk):
        def get_axis_features(chunk, axis):
            axis_min, axis_max, axis_mean = chunk[:, axis].min(), chunk[:, axis].max(), chunk[:, axis].mean()
            axis_median, axis_std = np.median(chunk[:, axis]), chunk[:, axis].std()

            return [
                axis_min, axis_max, axis_mean, axis_median, axis_std,
            ]

        output = chunk.flatten().tolist()
        output.extend(get_axis_features(chunk, axis=0))
        output.extend(get_axis_features(chunk, axis=1))

        return np.array(output)

    def predict(self, queue):
        data = np.array(queue)
        data = data[-self.length:]
        data = self.process_chunk(data).reshape((1, -1))
        prediction = self.model.predict(data).flatten()[:2]
        return prediction
