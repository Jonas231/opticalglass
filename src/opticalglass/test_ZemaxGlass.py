# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:24:31 2022

Use this script to create xlsx from .agf
@author: Jonas
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")

# https://github.com/nzhagen/zemaxglass
# install with python install setup.py

import ZemaxGlass as zg

glasslib = zg.ZemaxGlassLibrary()

glasslib.plot_dispersion('N-BK7', 'schott')

glasslib.plot_dispersion('SF11', 'schott', polyfit=True, fiterror = True)

#glasslib.plot_catalog_property_diagram("all", prop1 = "vd", prop2 = 'nd')

#glasslib.plot_dispersion('IRG22', 'SCHOTT_IRG_ZEMAX_July_2018')


def return_parse_glass_file(filename, printf = 0):
    '''
    Read a Zemax glass file (*.agf') and return its contents as a Python dictionary.

    Parameters
    ----------
    filename : str
        The file to parse.

    Returns
    -------
    glass_catalog : dict
        The dictionary containing glass data for all classes in the file.
    '''

    f = open(filename, 'r', encoding='latin1')
    glass_catalog = {}
    
    a = 0
    glassname_L=[]
    line_L = []
    for line in f:
        line_L.append(line)
        
        if printf and a == 1:
            print(line)
        if not line.strip(): continue
        #else: print("error!")
        
        if line.startswith('CC '): continue   # comment
        if line.startswith('NM '):            # glass name

            nm = line.split()
            print(nm)
            glassname = nm[1]
            glassname_L.append(glassname)
                
            glass_catalog[glassname] = {}
            glass_catalog[glassname]['dispform'] = int(nm[2])
            glass_catalog[glassname]['nd'] = float(nm[4])
            glass_catalog[glassname]['ne'] = np.NaN
            glass_catalog[glassname]['vd'] = float(nm[5])
            glass_catalog[glassname]['ve'] = np.NaN
            glass_catalog[glassname]['exclude_sub'] = 0 if (len(nm) < 7) else int(nm[6])
            glass_catalog[glassname]['status'] = 0 if (len(nm) < 8) else int(nm[7])
            glass_catalog[glassname]['meltfreq'] = 0 if ((len(nm) < 9) or (nm.count('-') > 0)) else int(nm[8])
            
        
        elif line.startswith('ED '):
            ed = line.split()
            glass_catalog[glassname]['tce'] = float(ed[1])
            glass_catalog[glassname]['density'] = float(ed[3])
            glass_catalog[glassname]['dpgf'] = float(ed[4])
            glass_catalog[glassname]['ignore_thermal_exp'] = 0 if (len(ed) < 6) else int(ed[5])
        elif line.startswith('CD '):
            cd = line.split()[1:]
            glass_catalog[glassname]['cd'] = [float(a) for a in cd]
            #print(cd)
            if  glass_catalog[glassname]['dispform'] == 2:
                glass_catalog[glassname]['B1'] = float(cd[0])
                glass_catalog[glassname]['C1'] = float(cd[1])
                glass_catalog[glassname]['B2'] = float(cd[2])
                glass_catalog[glassname]['C2'] = float(cd[3])
                glass_catalog[glassname]['B3'] = float(cd[4])
                glass_catalog[glassname]['C3'] = float(cd[5])
            
        elif line.startswith('TD '):
            td = line.split()[1:]
            print(td)
            if not td: continue     ## the Schott catalog sometimes uses an empty line for the "TD" label
            glass_catalog[glassname]['td'] = [float(a) for a in td]
            glass_catalog[glassname]['D0'] =  float(td[0])
            glass_catalog[glassname]['D1'] =  float(td[1])
            glass_catalog[glassname]['D2'] =  float(td[2])
            glass_catalog[glassname]['E0'] =  float(td[3])
            glass_catalog[glassname]['E1'] =  float(td[4])
            glass_catalog[glassname]['lambda'] =  float(td[5])
            
            
        elif line.startswith('OD '):
            od = line.split()[1:]
            od = zg.string_list_to_float_list(od)
            glass_catalog[glassname]['relcost'] = od[0]
            glass_catalog[glassname]['cr'] = od[1]
            glass_catalog[glassname]['fr'] = od[2]
            glass_catalog[glassname]['sr'] = od[3]
            glass_catalog[glassname]['ar'] = od[4]
            if (len(od) == 6):
                glass_catalog[glassname]['pr'] = od[5]
            else:
                glass_catalog[glassname]['pr'] = -1.0
        elif line.startswith('LD '):
            ld = line.split()[1:]
            glass_catalog[glassname]['ld'] = [float(a) for a in ld]
        elif line.startswith('IT '):
            it = line.split()[1:]
            #print(it)
            it_row = [float(a) for a in it]
            #print(it_row)
            if ('it' not in glass_catalog[glassname]):
                glass_catalog[glassname]['IT'] = {}
            #glass_catalog[glassname]['IT']['wavelength'] = it_row[0]
            #glass_catalog[glassname]['IT']['transmission'] = it_row[1]
            
            if len(it_row) > 2:
                glass_catalog[glassname]['TAUI'+str(it_row[2])+'/'+str(int(it_row[0]))] = it_row[1]
            #else:
            #    glass_catalog[glassname]['TAUI'+str(it_row[2])+'/'+str(int(it_row[0]))] = it_row[1]

            if len(it_row) > 2:
                glass_catalog[glassname]['IT']['thickness'] = it_row[2]
            else:
                glass_catalog[glassname]['IT']['thickness'] = np.NaN

    f.close()
    glass_catalog_df = pd.DataFrame.from_dict(glass_catalog, orient = "index")

    return glass_catalog, glassname_L, glass_catalog_df
           
# convert to Dataframe
file_n = r"C:\D\.spyder.python3\pyrateoptics\AGF_files\SCHOTT_IRG_ZEMAX_July_2018.agf"
#file_n = r"C:\D\.spyder.python3\pyrateoptics\AGF_files\infrared_2022.agf"

import codecs
def convertUTF16ToANSI(oldfile,newfile):
 
    #Open UTF8 text file
    f = codecs.open(oldfile,'r','utf16')
    utfstr = f.read()
    f.close()
    
         #Transcode UTF8 strings into ANSI strings
    outansestr = utfstr.encode('mbcs')
 
         #Save the transcoded text in binary format
    f = open(newfile,'wb')
    f.write(outansestr)
    f.close()

import os
filename_conv = os.path.join(os.path.dirname(file_n),os.path.split(file_n)[-1][:-4]+"_mod"+".agf")
convertUTF16ToANSI(file_n,filename_conv)


file_n2 = r"C:\D\.spyder.python3\pyrateoptics\AGF_files\schott.agf"
file_n2 = r"C:\D\.spyder.python3\pyrateoptics\AGF_files\infrared_2022.agf"
#file_n = file_n2

data, gn, data_df = return_parse_glass_file(filename = filename_conv, printf = 1)
data2, gn2, data_df2 = return_parse_glass_file(filename = file_n2, printf = 0)

#import pandas as pd

# write dataframe
#data_df2 = pd.DataFrame.from_dict(data2, orient = "index")

# write to xlsx from dataframe
data_df.to_excel(os.path.split(file_n)[-1][:-4]+".xlsx")  
data_df2.to_excel(os.path.split(file_n2)[-1][:-4]+".xlsx")

# calculate instantaneous dispersion
# see: https://oeosc.org/OP1%20Meeting%20Documents/2013/TF6/DSS%20IRSWG-Carlie.pdf
# get a coefficient from dataframe
# data_df.loc['IRG22']['B1']

def n_wavl(dispform, w, cd):
    if (dispform == 0):
        ## use this for AIR and VACUUM
        pass
    elif (dispform == 1):
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 2):  ## Sellmeier1
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + (cd[4] * w**2 / (w**2 - cd[5]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 3):  ## Herzberger
        L = 1.0 / (w**2 - 0.028)
        indices = np.cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + (cd[4] * w**4) + (cd[5] * w**6)
    elif (dispform == 4):  ## Sellmeier2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + (cd[3] / (w**2 - (cd[4])**2))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 5):  ## Conrady
        indices = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
    elif (dispform == 6):  ## Sellmeier3
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                      (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 7):  ## HandbookOfOptics1
        formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 8):  ## HandbookOfOptics2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 9):  ## Sellmeier4
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + (cd[3] * w**2 / (w**2 - cd[4]))
        indices = np.sqrt(formula_rhs)
    elif (dispform == 10):  ## Extended1
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                      (cd[5] * w**-8) + (cd[6] * w**-10) + (cd[7] * w**-12)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 11):  ## Sellmeier5
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                      (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                      (cd[8] * w**2 / (w**2 - cd[9]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 12):  ## Extended2
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                      (cd[5] * w**-8) + (cd[6] * w**4) + (cd[7] * w**6)
        indices = np.sqrt(formula_rhs)
    #else:
    #    raise ValueError('Dispersion formula #' + str(dispform) + ' (for glass=' + glass + ' in catalog=' + catalog + ') is not a valid choice.')

    return formula_rhs

def n_wavl_1st_deriv(dispform):
    import sympy as sp
    ws = sp.Symbol("ws")
    if (dispform == 0):
        ## use this for AIR and VACUUM
        pass
    elif (dispform == 1):
        deriv_f = 0*cd[0] + (2*cd[1] * w**1) + (-2*cd[2] * w**-1) + (-4*cd[3] * w**-3) + (-6*cd[4] * w**-5) + (-8*cd[5] * w**-7)

    elif (dispform == 2):  ## Sellmeier1
        cds = [sp.Symbol("B1"),sp.Symbol( "C1"),sp.Symbol( "B2"), sp.Symbol("C2"), sp.Symbol("B3"), sp.Symbol("C3")]
        formula_rhs_s = (cds[0] * ws**2 / (ws**2 - cds[1])) + (cds[2] * ws**2 / (ws**2 - cds[3])) + (cds[4] * ws**2 / (ws**2 - cds[5]))
        #print(formula_rhs_s)
        indices = sp.sqrt(formula_rhs_s + 1)


    elif (dispform == 3):  ## Herzberger
        L = 1.0 / (w**2 - 0.028)
        indices = np.cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + (cd[4] * w**4) + (cd[5] * w**6)
    elif (dispform == 4):  ## Sellmeier2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + (cd[3] / (w**2 - (cd[4])**2))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 5):  ## Conrady
        indices = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
    elif (dispform == 6):  ## Sellmeier3
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                      (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 7):  ## HandbookOfOptics1
        formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 8):  ## HandbookOfOptics2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 9):  ## Sellmeier4
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + (cd[3] * w**2 / (w**2 - cd[4]))
        indices = np.sqrt(formula_rhs)
    elif (dispform == 10):  ## Extended1
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                      (cd[5] * w**-8) + (cd[6] * w**-10) + (cd[7] * w**-12)
        indices = np.sqrt(formula_rhs)
    elif (dispform == 11):  ## Sellmeier5
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                      (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                      (cd[8] * w**2 / (w**2 - cd[9]))
        indices = np.sqrt(formula_rhs + 1.0)
    elif (dispform == 12):  ## Extended2
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                      (cd[5] * w**-8) + (cd[6] * w**4) + (cd[7] * w**6)
       
        indices = np.sqrt(formula_rhs)
    #else:
    #    raise ValueError('Dispersion formula #' + str(dispform) + ' (for glass=' + glass + ' in catalog=' + catalog + ') is not a valid choice.')
    	
    deriv = sp.diff(indices, ws)
    #deriv = deriv.evalf(subs={x: 3.14})
    
    
    return indices, deriv, cds

formula_rhs_s,deriv, cds = n_wavl_1st_deriv(dispform =2)

def fill_deriv_values(data_df,deriv, cds, glasstype = 'IRG22'):


    #glasstype = 'IRG22'
    cd = data_df.loc[glasstype]['cd']
    for i in range(0, len(cds)):
        deriv = deriv.evalf(subs={cds[i]: cd[i+1]})
        
    minwl = data_df.loc[glasstype]['ld'][0]
    maxwl = data_df.loc[glasstype]['ld'][1]
    wavl = np.linspace(minwl, maxwl, num = 300)
    from sympy import lambdify
    import sympy as spy
    ws = spy.Symbol("ws")
    f = lambdify(ws, deriv, "numpy")
    deriv_r = f(wavl)
    
    fig0 = plt.figure("inst")
    ax = plt.subplot(111)
    plt.plot(wavl, deriv_r)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    
    return deriv_r

deriv_r = fill_deriv_values(data_df, deriv, cds, glasstype = 'IRG22')
    


    
    
#deriv = fill_deriv_values(deriv, w  = np.linspace(1,10, 100), cd = data_df.loc['IRG22']['cd'][:6], cds = cds)


def det_disp(data_df, glasstype = 'IRG22'):
    minwl = data_df.loc[glasstype]['ld'][0]
    maxwl = data_df.loc[glasstype]['ld'][1]
    wavl = np.linspace(minwl, maxwl, num = 30)
    
    

    B1 = data_df.loc[glasstype]['B1']
    B2 = data_df.loc[glasstype]['B2']
    B3 = data_df.loc[glasstype]['B3']
    C1 = data_df.loc[glasstype]['C1']
    C2 = data_df.loc[glasstype]['C2']
    C3 = data_df.loc[glasstype]['C3']
    cd = data_df.loc[glasstype]['cd']
    coeffs = np.array([B1, B2, B3, C1, C2, C3])
    dispform = data_df.loc[glasstype]['dispform']
    print("dispform: ", dispform)
    n = n_wavl(dispform, wavl, cd)
    # perform fit with Sellmeier formula
    fig = plt.figure("dispersion func")
    plt.plot(wavl,n)
    
    # display formula
    
    
    #n =  np.sqrt(1+ B1*wavl**2 /(wavl**2-C1)  + B2*wavl**2 /(wavl**2-C2) + B3*wavl**2 /(wavl**2-C3) )    # Sellmeier formula
    nu = 1/n * ( -B1*C1*wavl/(wavl-C1)**2  -B2*C2*wavl/(wavl-C2)**2  -B3*C3*wavl/(wavl-C3)**2)
    return wavl, nu, coeffs

wavl, nu_test, coeffs = det_disp(data_df2, glasstype = 'GERMANIUM')
#wavl, nu_test = det_disp(data_df2, glasstype = 'N-BK7')

fig_inst = plt.figure("dispersion")
ax = plt.subplot(111)
ax.plot(wavl, nu_test)
#ax.set_xscale('log')
#ax.set_yscale('log')




# https://opticalglass.readthedocs.io/en/latest/OG_Quickstart/OG_Quickstart.html

import opticalglass as og
import opticalglass.glassmap as gm
from opticalglass.glassfactory import create_glass




#plt.close("all")
bk7 = create_glass('N-BK7', 'Schott')
print(bk7)

wl = np.linspace(365., 700., num=75)
rn = bk7.calc_rindex(wl)
plt.figure("opticalglass")
plt.plot(wl,rn)

t_data = bk7.transmission_data()
plt.figure("opticalglass, transmission")
plt.plot(np.array(t_data)[:,0],np.array(t_data)[:,1])


gmf = plt.figure(FigureClass=gm.GlassMapFigure,
                 glass_db=gm.GlassMapDB()).plot()


