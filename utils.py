import numpy as np

### Useful functions ###

def is_exist(text, word):
    try:
        if text.index(word) >= 0:
            return True
    except ValueError:
        return False

def dictToArray(dict):
    data = list(dict.items())
    return np.array(data)




