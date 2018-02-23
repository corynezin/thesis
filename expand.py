def expand_number(m):
    word = []
    for char in m.group(0):
        word.append(digits_dictionary[char])
    return ' '.join(word) + ' '

def expand_contraction(m): 
    return contractions_dictionary[m.group(0)]
