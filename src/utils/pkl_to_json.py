import pickle
import argparse
import json
import numpy as np

def convert_to_json(data_file, output_path):
    # Load the data
    with open(data_file, "rb") as in_file:
        data = pickle.load(in_file)

    # custom json encoder to handle numpy data types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
        
    # dump the data to json for easier inspection
    with open(output_path, "w") as out_file:
        json.dump(data, out_file, cls=NumpyEncoder, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_file",
        type=str,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debug_data.json"
    )
    args = parser.parse_args()
    pkl_path = args.pkl_file
    convert_to_json(pkl_path)