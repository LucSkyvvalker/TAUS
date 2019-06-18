
# find mismatches in brackets or accolades
def getMismatch(source, target):
    tgtBrack = 0
    tgtAccolade = 0
    srcBrack = 0
    srcAccolade = 0
    # check for mismatches in target sentence
    for char in str(target):
        if char == '(':
            tgtBrack = 1
        if tgtBrack == 1 and char == ')':
            tgtBrack = 0
        if char == '"':
            tgtAccolade = 1
        if tgtAccolade == 1 and char == '"':
            tgtAccolade = 0
    # if mismatch, check if it was present in source
    if tgtBrack == 1 or tgtAccolade == 1:
        for char in str(source):
            if char == '(':
                srcBrack = 1
            if srcBrack == 1 and char == ')':
                srcBrack = 0
            if char == '"':
                srcAccolade = 1
            if srcAccolade == 1 and char == '"':
                srcAccolade = 0
        # if present in src, mismatch is false
        if srcBrack == 1 and tgtBrack == 1:
            tgtBrack = 0
        if srcAccolade == 1 and tgtAccolade == 1:
            tgtAccolade = 0
    return (tgtBrack+tgtAccolade)

