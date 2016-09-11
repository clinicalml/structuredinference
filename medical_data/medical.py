"""
Utilities for medical data
"""

def cleanLabels(patdict):
    """
    Clean patient labels
    """
    newdict = []
    oldval  = {}
    oldval['age'] = 18
    oldval['a1c'] = 0
    oldval['gluc']= 0
    
    realname = {}
    realname['gluc'] = 'Gluc.'
    realname['a1c'] = 'A1c'
    realname['age'] = 'Age'
    for idx,data in enumerate(patdict):
        if data[-1]=='n':
            data = data[:-1]
        data = data.replace('DMI ','Diabetes ')
        data = data.replace('DMII ','Diabetes ')
        data = data.replace('Chr ','Chronic ')
        data = data.replace('Crnry ','Coronary ')
        data = data.replace('Cor ','Coronary ')
        data = data.replace('CHF ','Congestive Heart Failure ')
        data = data.replace('Chronic ischemic hrt dis NOS','Chronic ischemic heart disease, unspecified')
        data = data.replace('&','and')
        data = data.replace('isF','is Female')
        if 'age_' in data or 'a1c_' in data or 'gluc_' in data:
            if 'age' in data:
                val = 'age'
            elif 'a1c' in data:
                val = 'a1c'
            elif 'gluc' in data:
                val = 'gluc'
            else:
                assert False,'error'
            newval = float(data.split('lt')[1].replace('_',''))
            newdata= str(oldval[val])+' $<$ '+realname[val]+' $<$ '+str(newval)
            oldval[val] = newval
        else:
            newdata = data.replace('_',' ')
        newdict.append(newdata)
    return newdict
