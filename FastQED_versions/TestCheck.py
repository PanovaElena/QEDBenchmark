# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:58:46 2023

@author: bashi
"""

import os
import re
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from scipy.optimize import curve_fit


LV = 2.99792458e10
Me = 0.91093837e-27
Qe = 4.8032e-10

OUT_NUM = 5 # plot using the 5th data

def InputDictionary(pathToInput,dictInput):
    regString=r"""Setting (?:variable|string)\S*\s*(?P<nameVar>\S*)\s*=\s*(?P<valueVar>\S+)\s*"""
    regExpr=re.compile(regString)
    inpFile=open(pathToInput,'r')
    
    for line in inpFile:
        foundVar=regExpr.search(line)
        if foundVar:
            curDict=foundVar.groupdict()
            dictInput[curDict['nameVar']]=curDict['valueVar']
    inpFile.close()
    return 0

def TakeParam(pathToInput,param):
    regString=r"Setting (?:variable|string)\S*\s*%s\s*=\s*(?P<valueVar>\S+)\s*" % param
    regExpr=re.compile(regString)
    inpFile=open(pathToInput,'r')
    for line in inpFile:
        foundVar=regExpr.search(line)
        if foundVar:
            curDict=foundVar.groupdict()
            valParam=curDict['valueVar']
            break
    inpFile.close()
    return valParam

def NumberFiles(pathtodir):
    return len([name for name in os.listdir(pathtodir) if os.path.isfile(pathtodir+name)])

def AppF(timeseries,a,b):
    return a*np.exp(b*timeseries)

nargv = len(sys.argv) - 1
colors=['red', 'blue', 'green', 'orange', 'magenta', 'yellow', 'pink', 'cyan', 'brown', 'grey', 'purple', 'olive']
  
fig = plt.figure(figsize=(15,5.5))
canvas = FigureCanvas(fig)
ax1=fig.add_subplot(1, 5, 1) #fig.add_axes([0.03,0.15,0.28,0.8])
ax1.set_yscale('log')
ax2=fig.add_subplot(1, 5, 2) #fig.add_axes([0.35,0.15,0.28,0.8])
ax3=fig.add_subplot(1, 5, 3) #fig.add_axes([0.68,0.15,0.28,0.8])
ax4=fig.add_subplot(1, 5, 4)
ax5=fig.add_subplot(1, 5, 5)

nsims=nargv
moments=[]
for isim in range(nsims):
    path = sys.argv[isim+1]
    moments.append(NumberFiles(path+'/BasicOutput/data/Electron1Dx/')-1)

tmom = OUT_NUM

lines = [[] for el in range(nsims)]

for isim in range(nsims):
    path = sys.argv[isim+1]
    #tmom = int(sys.argv[2*isim+2])
    inputDict={}
    InputDictionary(path+'/ParsedInput.txt', inputDict)
    wavlen=float(inputDict['Wavelength'])
    xmin=float(inputDict['Electron1Dx.SetBounds_0'])
    xmax=float(inputDict['Electron1Dx.SetBounds_1'])
    
    
    stepsperiod=int(inputDict['StepsPerPeriod'])
    boiter=int(inputDict['BOIterationPass'])
    
    numparts=np.zeros(tmom+1)
    tnumparts=np.linspace(0,tmom*boiter//stepsperiod,tmom+1)
    for imom in range(tmom+1):
        numparts[imom]+=np.loadtxt(path+'/BasicOutput/data/Electron1Dx/%.6d.txt' % imom).sum()
    np.savetxt(path+"/tnumparts.txt",tnumparts)
    np.savetxt(path+"/numparts.txt",numparts)
    ax1.plot(tnumparts,numparts,color=colors[isim],linestyle="--",linewidth=1,label=sys.argv[isim + 1])
    params, pcov = curve_fit(AppF,tnumparts[(tmom-stepsperiod//boiter//2*3):],numparts[(tmom-stepsperiod//boiter//2*3):])
    approximation=AppF(tnumparts,params[0],params[1])
    ax1.plot(tnumparts,approximation,color='k',linestyle=':',linewidth=0.5)
    #ax1.text(0.3, 0.94-isim*0.05, r'$\Gamma_{%d}T = %.6g$' % (isim,params[1]), fontsize=14,transform=ax1.transAxes)
    
    distrib_el=np.loadtxt(path+'/BasicOutput/data/Electron1Dx/%.6d.txt' % tmom)
    distrib_pos=np.loadtxt(path+'/BasicOutput/data/Positron1Dx/%.6d.txt' % tmom)
    distrib_ph=np.loadtxt(path+'/BasicOutput/data/Photon1Dx/%.6d.txt' % tmom)
    msizex=distrib_el.size
    quart_msizex=msizex#//4
    distrib_res=np.zeros(quart_msizex)
    distrib_resph=np.zeros(quart_msizex)
    distrib_res[:]=distrib_el[0:quart_msizex]+distrib_pos[0:quart_msizex]#+distrib_el[(2*quart_msizex):(3*quart_msizex)]+distrib_pos[(2*quart_msizex):(3*quart_msizex)]
    #distrib_res[:]+=distrib_el[(2*quart_msizex-1):(quart_msizex-1):-1]+distrib_pos[(2*quart_msizex-1):(quart_msizex-1):-1]
    #distrib_res[:]+=distrib_el[(4*quart_msizex-1):(3*quart_msizex-1):-1]+distrib_pos[(4*quart_msizex-1):(3*quart_msizex-1):-1]
    #distrib_res[:]/=8
    
    distrib_resph=distrib_ph[0:quart_msizex]#+distrib_ph[(2*quart_msizex):(3*quart_msizex)]
    #distrib_resph[:]+=distrib_ph[(2*quart_msizex-1):(quart_msizex-1):-1]+distrib_ph[(4*quart_msizex-1):(3*quart_msizex-1):-1]
    #distrib_resph[:]/=4

    #xax=np.linspace(0,(xmax-xmin)/(4*wavlen),quart_msizex,False)
    xax=np.linspace(0,(xmax-xmin),quart_msizex,False)
    np.savetxt(path+'/distrib_res.txt',distrib_res)
    np.savetxt(path+'/distrib_resph.txt',distrib_resph)
    np.savetxt(path+'/xax.txt',xax)
    xax+=0.5*(xmax-xmin)/msizex
    lines[isim].append(ax2.plot(xax,distrib_res,color=colors[isim],linestyle="--",linewidth=1, label=sys.argv[isim + 1])[0])
    lines[isim].append(ax3.plot(xax,distrib_resph,color=colors[isim],linestyle="--",linewidth=1, label=sys.argv[isim + 1])[0])
    
    ey = np.loadtxt(path+'/BasicOutput/data/Ey1D/%.6d.txt' % tmom)
    ez = np.loadtxt(path+'/BasicOutput/data/Ez1D/%.6d.txt' % tmom)
    by = np.loadtxt(path+'/BasicOutput/data/By1D/%.6d.txt' % tmom)
    bz = np.loadtxt(path+'/BasicOutput/data/Bz1D/%.6d.txt' % tmom)
    lines[isim].append(ax4.plot(xax, ey,color=colors[isim],linestyle="--",linewidth=1, label=sys.argv[isim + 1])[0])
    lines[isim].append(ax4.plot(xax, ez,color=colors[isim],linestyle="--",linewidth=1)[0])
    lines[isim].append(ax4.plot(xax, by,color=colors[isim],linestyle="--",linewidth=1)[0])
    lines[isim].append(ax4.plot(xax, bz,color=colors[isim],linestyle="--",linewidth=1)[0])
    
    energy = np.loadtxt(path+'/BasicOutput/data/PhotonEnergy1Dx/%.6d.txt' % tmom)
    lines[isim].append(ax5.plot(xax, energy,color=colors[isim],linestyle="--",linewidth=1, label=sys.argv[isim + 1])[0])
        
ax1.legend(loc=0,frameon=False,fontsize=9)
ax2.legend(loc=0,frameon=False,fontsize=9)
ax3.legend(loc=0,frameon=False,fontsize=9)
ax4.legend(loc=0,frameon=False,fontsize=9)

ax1.set_title("particles(t)")
ax2.set_title("charged paricles")
ax3.set_title("photons")
ax4.set_title("field")
ax5.set_title("photon energy")

#ax2.set_ylim((1e-3, 1e3))
#ax3.set_ylim((1e-3, 1e4))
#ax5.set_ylim((1e-3, 1e1))
#
#ax2.semilogy()
#ax3.semilogy()
#ax5.semilogy()

def animate(tmom):
    for isim in range(nsims):
        path = sys.argv[isim+1]
        #tmom = int(sys.argv[2*isim+2])
        inputDict={}
        InputDictionary(path+'/ParsedInput.txt', inputDict)
        wavlen=float(inputDict['Wavelength'])
        xmin=float(inputDict['Electron1Dx.SetBounds_0'])
        xmax=float(inputDict['Electron1Dx.SetBounds_1'])
        
        iline = 0
        
        stepsperiod=int(inputDict['StepsPerPeriod'])
        boiter=int(inputDict['BOIterationPass'])
        
        numparts=np.zeros(tmom+1)
        tnumparts=np.linspace(0,tmom*boiter//stepsperiod,tmom+1)
        for imom in range(tmom+1):
            numparts[imom]+=np.loadtxt(path+'/BasicOutput/data/Electron1Dx/%.6d.txt' % imom).sum()
        np.savetxt(path+"/tnumparts.txt",tnumparts)
        np.savetxt(path+"/numparts.txt",numparts)
        #ax1.plot(tnumparts,numparts,color=colors[isim],linestyle="--",linewidth=1,label=sys.argv[isim + 1])
        #params, pcov = curve_fit(AppF,tnumparts[(tmom-stepsperiod//boiter//2*3):],numparts[(tmom-stepsperiod//boiter//2*3):])
        #approximation=AppF(tnumparts,params[0],params[1])
        #ax1.plot(tnumparts,approximation,color='k',linestyle=':',linewidth=0.5)
        #ax1.text(0.3, 0.94-isim*0.05, r'$\Gamma_{%d}T = %.6g$' % (isim,params[1]), fontsize=14,transform=ax1.transAxes)
        
        distrib_el=np.loadtxt(path+'/BasicOutput/data/Electron1Dx/%.6d.txt' % tmom)
        distrib_pos=np.loadtxt(path+'/BasicOutput/data/Positron1Dx/%.6d.txt' % tmom)
        distrib_ph=np.loadtxt(path+'/BasicOutput/data/Photon1Dx/%.6d.txt' % tmom)
        msizex=distrib_el.size
        quart_msizex=msizex#//4
        distrib_res=np.zeros(quart_msizex)
        distrib_resph=np.zeros(quart_msizex)
        distrib_res[:]=distrib_el[0:quart_msizex]+distrib_pos[0:quart_msizex]#+distrib_el[(2*quart_msizex):(3*quart_msizex)]+distrib_pos[(2*quart_msizex):(3*quart_msizex)]
        #distrib_res[:]+=distrib_el[(2*quart_msizex-1):(quart_msizex-1):-1]+distrib_pos[(2*quart_msizex-1):(quart_msizex-1):-1]
        #distrib_res[:]+=distrib_el[(4*quart_msizex-1):(3*quart_msizex-1):-1]+distrib_pos[(4*quart_msizex-1):(3*quart_msizex-1):-1]
        #distrib_res[:]/=8
        
        distrib_resph=distrib_ph[0:quart_msizex]#+distrib_ph[(2*quart_msizex):(3*quart_msizex)]
        #distrib_resph[:]+=distrib_ph[(2*quart_msizex-1):(quart_msizex-1):-1]+distrib_ph[(4*quart_msizex-1):(3*quart_msizex-1):-1]
        #distrib_resph[:]/=4
	
        #xax=np.linspace(0,(xmax-xmin)/(4*wavlen),quart_msizex,False)
        xax=np.linspace(0,(xmax-xmin),quart_msizex,False)
        np.savetxt(path+'/distrib_res.txt',distrib_res)
        np.savetxt(path+'/distrib_resph.txt',distrib_resph)
        np.savetxt(path+'/xax.txt',xax)
        xax+=0.5*(xmax-xmin)/msizex
        lines[isim][iline].set_data(xax,distrib_res)
        iline += 1
        lines[isim][iline].set_data(xax,distrib_resph)
        iline += 1
        
        ey = np.loadtxt(path+'/BasicOutput/data/Ey1D/%.6d.txt' % tmom)
        ez = np.loadtxt(path+'/BasicOutput/data/Ez1D/%.6d.txt' % tmom)
        by = np.loadtxt(path+'/BasicOutput/data/By1D/%.6d.txt' % tmom)
        bz = np.loadtxt(path+'/BasicOutput/data/Bz1D/%.6d.txt' % tmom)
        lines[isim][iline].set_data(xax, ey)
        iline += 1
        lines[isim][iline].set_data(xax, ez)
        iline += 1
        lines[isim][iline].set_data(xax, by)
        iline += 1
        lines[isim][iline].set_data(xax, bz)
        iline += 1
        
        energy = np.loadtxt(path+'/BasicOutput/data/PhotonEnergy1Dx/%.6d.txt' % tmom)
        lines[isim][iline].set_data(xax, energy)

animate(tmom)
canvas.print_figure('Compare.png', dpi=400)
canvas.print_figure('GadtN.pdf', dpi=400)

#ani = animation.FuncAnimation(fig, animate, blit=False, interval=100, repeat=True)
#plt.show()

