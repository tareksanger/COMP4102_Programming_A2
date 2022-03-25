import numpy as np
import pandas as pd
import os

def save_model(model, history, dir):
    pd.DataFrame(history.history).to_hdf(os.path.join(dir, f"history.h5"), 'history')
    model.save(os.path.join(dir, f'/model.h5'))
    model.save_weights(os.path.join(dir, f'/final_weights.hdf5'), overwrite=True)
