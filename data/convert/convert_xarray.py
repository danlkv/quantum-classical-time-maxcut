import zlib
import xarray
import glob
import json
import pickle
import os

def main():
    # Print the running directory
    print("Running in directory: {}".format(os.getcwd()))
    files = glob.glob('./*.pkl')

    for file in files:
        print("Loading {}".format(file))
        try:
            ds = pickle.load(open(file, 'rb'))
        except Exception as e:
            print("Error: {}".format(e))
            continue

        # convert to json
        dic = ds.to_dict()
        comp = zlib.compress(json.dumps(dic).encode('utf-8'))
        # save to file
        with open(file.replace('.pkl', '.json.zlib'), 'wb') as f:
            f.write(comp)
        print("Saved {}".format(file.replace('.pkl', '.json.gz')))



if __name__ == '__main__':
    main()
