#%%
#%matplotlib inline

#Test

import os

os.environ['PROJ_LIB'] = '/Applications/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj'

# Importation des modules utiles pour les TPs
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from scipy.stats import linregress
import matplotlib.colors as colors


def spatialmean(V,lat):		# lat: vecteur 1D des lattitudes
    R = np.mean(V,2) 		# Moyenne sur les longitudes (non pondérée)
    T = len(R[:,1])		    # Nombre de pas de temps pour la variable
    coslat = np.cos(np.pi*lat/180)
    Rtemp = np.zeros((T,len(lat)))
    for t in range(0,T):	# Moyenne sur les lattitudes (pondérée par le cosinus de la lattitude)
        Rtemp[t,:] = [x*y for x, y in zip(R[t,:],coslat)]
        Rtemp[t,:] = Rtemp[t,:]/sum(coslat)
    R = np.sum(Rtemp,1)		# Retourne R, la variable V moyennée spatiallement en fonction du temps (R est un vecteur 1D)
    return R


# Dark graph
plt.style.use('dark_background')
#plt.style.use('default')

##### Récupération et observation des variables des fichiers netcdf #####
# Récupération des données du fichier fichier.nc dans la structure data
data = Dataset("/Users/felixlangot/Google Drive (felixlangot@gmail.com)/UVSQ/Modeles_pour_l'etude_des_climats/TP_iLoveClim/felixlangot-CTRL-YR.nc")
data2 = Dataset("/Users/felixlangot/Google Drive (felixlangot@gmail.com)/UVSQ/Modeles_pour_l'etude_des_climats/TP_iLoveClim/felixlangot-1pcCO2-YR.nc")
# On regarde les variables contenues dans data et la variable variable
print("data:")
print(data)
print(data.variables['ts'])
# On créer de nouvelles variables à partir des données pour les manipuler plus facilement dans Python

Ts = data.variables['ts'][:]

lon = data.variables['lon'][:]
lat = data.variables['lat'][:]

############################# TEMPERATURE 1D ###################################
print('#######################################################################')
print('Evolution de la temperature a la surface en (0,0) sur 1000 ans')
print('#######################################################################')

Tsloc = Ts[:,0,0]

t = np.arange(0,1000)

plt.plot(t,Tsloc,'b-', linewidth=1, label='legende')	# Plot 1D de z en fonction de t en bleu avec une légende
#plt.title('Titre', fontsize=20)					# Titre
plt.xlabel('$t$ (unité de t)', fontsize=20)			# Axe des abscisses
plt.ylabel('$T$ (K)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
#plt.savefig('figure.eps', format='eps', dpi=600)		# Sauvegarder la figure au format eps
plt.show()

print('#######################################################################')
print('Evolution de la temperature moyenne a la surface sur 1000 ans')
print('#######################################################################')

Tsevol = spatialmean(Ts, lat)

plt.plot(t,Tsevol,linewidth=1)
plt.title('Temperature with np.mean loop', fontsize=20)					# Titre
plt.xlabel('$t$ (yr)', fontsize=20)			# Axe des abscisses
plt.ylabel('$T$ (K)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
plt.savefig('../Graph1D/EvolCTRL.eps', format='eps', dpi=600)		# Sauvegarder la figure au format eps
plt.show()

############################# TEMPERATURE 2D ###################################

print('#######################################################################')
print('Evolution de la temperature 2D')
print('#######################################################################')
GifQ = input('Do you want to create a .gif ? (y or n) ')

if GifQ == 'y':
    for i in np.arange(0,1000):

        map = Basemap(projection='merc',llcrnrlon=-180.,llcrnrlat=-80.,
                  urcrnrlon=180.,urcrnrlat=80.,resolution='c')

        lons,lats= np.meshgrid(lon-180,lat)

        x,y = map(lons,lats)
        # Carte de la variable
        temp = map.pcolormesh(x,y,Ts[i,:,:],cmap='coolwarm')
        map.drawcoastlines() 			# lignes de côte
        paralleles = [-60, -30, 0, 30, 60]
        meridiens = [-120, -60, 0, 60, 120]
        map.drawparallels(paralleles,labels=[True,False,False,False],fontsize=10) 	# parallèles et méridiens
        map.drawmeridians(meridiens,labels=[False,False,False,True],fontsize=10) 	# labels = [left,right,top,bottom]
        cmap = 'coolwarm'
        colormesh = map.pcolormesh(x, y, Ts[i,:,:], vmin = 220, vmax = 310, cmap=cmap)
        cb = map.colorbar(colormesh)
#    cb = map.colorbar(temp,"right", size="5%", pad="2%", vmin=220, vmax=310)
        cb.set_label('$T$ (K)', fontsize=20)
        plt.title('year ' + str(i), fontsize=20)
# Sauvegarde
        plt.savefig("../Graph2D/CTRL/Gif/Graph" + str(i) +".png", format='PNG')
        plt.show()

if GifQ == 'n':
    map = Basemap(projection='merc',llcrnrlon=-180.,llcrnrlat=-80.,
                  urcrnrlon=180.,urcrnrlat=80.,resolution='c')

    lons,lats= np.meshgrid(lon-180,lat)

    x,y = map(lons,lats)
    # Carte de la variable
    temp = map.pcolormesh(x,y,Ts[500,:,:],cmap='coolwarm')
    map.drawcoastlines() 			# lignes de côte
    paralleles = [-60, -30, 0, 30, 60]
    meridiens = [-120, -60, 0, 60, 120]
    map.drawparallels(paralleles,labels=[True,False,False,False],fontsize=10) 	# parallèles et méridiens
    map.drawmeridians(meridiens,labels=[False,False,False,True],fontsize=10) 	# labels = [left,right,top,bottom]
    cmap = 'coolwarm'
    colormesh = map.pcolormesh(x, y, Ts[500,:,:], vmin = 220, vmax = 310, cmap=cmap)
    cb = map.colorbar(colormesh)
#    cb = map.colorbar(temp,"right", size="5%", pad="2%", vmin=220, vmax=310)
    cb.set_label('$T$ (K)', fontsize=20)
    plt.title('year 500', fontsize=20)
# Sauvegarde
    plt.savefig("../Graph2D/CTRL/TempCTRL.png", format='PNG')
    plt.show()

# %%

#%%

######################## COMPARAISON TEMPERATURE ###############################
print('#######################################################################')
print('Temperature 1D 1pcCO2')
print('#######################################################################')

print(data.variables['ts'])
Ts1pcCO2 = data2.variables['ts'][:]

TsevolC = spatialmean(Ts1pcCO2, lat)
t2 = np.arange(0, 500)

plt.plot(t2, TsevolC, linewidth=1)
plt.title('Temperature 1pcCO2', fontsize=20)					# Titre
plt.xlabel('$t$ (yr)', fontsize=20)			# Axe des abscisses
plt.ylabel('$T$ (K)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
plt.savefig('../Graph1D/Evol1pcCO2.eps', format='eps', dpi=600)		# Sauvegarder la figure au format eps
plt.show()

#%%

#%%
Tdiff = TsevolC - np.mean(Tsevol[400:500])
plt.plot(t2, Tdiff, linewidth=1)
plt.title('Temperature difference', fontsize=20)					# Titre
plt.xlabel('$t$ (yr)', fontsize=20)			# Axe des abscisses
plt.ylabel('$T$ (K)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
plt.savefig('../Graph1D/Diff.eps', format='eps', dpi=600)		# Sauvegarder la figure au format eps
plt.show()

# %%   

#%%
print('#######################################################################')
print('Temperature 2D 1pcCO2')
print('#######################################################################')

GifQ = input('Do you want to create a .gif ? (y or n) ')

if GifQ == 'y':
    for i in np.arange(0, 500):

        map = Basemap(projection='merc', llcrnrlon=-180., llcrnrlat=-80.,
                      urcrnrlon=180., urcrnrlat=80., resolution='c')

        lons, lats = np.meshgrid(lon-180, lat)

        x, y = map(lons, lats)
        # Carte de la variable
        temp = map.pcolormesh(x, y, Ts1pcCO2[i, :, :], cmap='coolwarm')
        map.drawcoastlines() 			# lignes de côte
        paralleles = [-60, -30, 0, 30, 60]
        meridiens = [-120, -60, 0, 60, 120]
        # parallèles et méridiens
        map.drawparallels(paralleles, labels=[
                          True, False, False, False], fontsize=10)
        # labels = [left,right,top,bottom]
        map.drawmeridians(meridiens, labels=[
                          False, False, False, True], fontsize=10)
        cmap = 'coolwarm'
        colormesh = map.pcolormesh(
            x, y, Ts[i, :, :], vmin=220, vmax=310, cmap=cmap)
        cb = map.colorbar(colormesh)
#    cb = map.colorbar(temp,"right", size="5%", pad="2%", vmin=220, vmax=310)
        cb.set_label('$T$ (K)', fontsize=20)
        plt.title('year ' + str(i), fontsize=20)
# Sauvegarde
        plt.savefig("../Graph2D/1pcCO2/Gif/Graph" + str(i) + ".png", format='PNG')
        plt.show()

if GifQ == 'n':
    map = Basemap(projection='merc', llcrnrlon=-180., llcrnrlat=-80.,
                  urcrnrlon=180., urcrnrlat=80., resolution='c')

    lons, lats = np.meshgrid(lon-180, lat)

    x, y = map(lons, lats)
    # Carte de la variable
    temp = map.pcolormesh(x, y, Ts1pcCO2[250, :, :], cmap='coolwarm')
    map.drawcoastlines() 			# lignes de côte
    paralleles = [-60, -30, 0, 30, 60]
    meridiens = [-120, -60, 0, 60, 120]
    # parallèles et méridiens
    map.drawparallels(paralleles, labels=[
                      True, False, False, False], fontsize=10)
    # labels = [left,right,top,bottom]
    map.drawmeridians(meridiens, labels=[
                      False, False, False, True], fontsize=10)
    cmap = 'coolwarm'
    colormesh = map.pcolormesh(
        x, y, Ts1pcCO2[499, :, :], vmin=220, vmax=310, cmap=cmap)
    cb = map.colorbar(colormesh)
#    cb = map.colorbar(temp,"right", size="5%", pad="2%", vmin=220, vmax=310)
    cb.set_label('$T$ (K)', fontsize=20)
    plt.title('year 250', fontsize=20)
# Sauvegarde
    plt.savefig("../Graph2D/1pcCO2/Temp1pcCO2.png", format='PNG')
    plt.show()

#%%


#%%
print('#######################################################################')
print('Temperature moyenne sur 500 ans 2D')
print('#######################################################################')

TmCTRL = np.zeros((32,64))
for i in np.arange(0, len(lat)):
    for j in np.arange(0,len(lon)):
        TmCTRL[i,j] = np.mean(Ts[0:500,i,j])

map = Basemap(projection='merc', llcrnrlon=-180., llcrnrlat=-80.,
              urcrnrlon=180., urcrnrlat=80., resolution='c')

lons, lats = np.meshgrid(lon-180, lat)

x, y = map(lons, lats)
# Carte de la variable
temp = map.pcolormesh(x, y, TmCTRL, cmap='coolwarm')
map.drawcoastlines() 			# lignes de côte
paralleles = [-60, -30, 0, 30, 60]
meridiens = [-120, -60, 0, 60, 120]
# parallèles et méridiens
map.drawparallels(paralleles, labels=[
    True, False, False, False], fontsize=10)
# labels = [left,right,top,bottom]
map.drawmeridians(meridiens, labels=[
    False, False, False, True], fontsize=10)
cmap = 'coolwarm'
colormesh = map.pcolormesh(x, y, TmCTRL, cmap=cmap)
cb = map.colorbar(colormesh)
#    cb = map.colorbar(temp,"right", size="5%", pad="2%", vmin=220, vmax=310)
cb.set_label('$T$ (K)', fontsize=20)
plt.title('Moyenne 2D 500 ans', fontsize=20)
# Sauvegarde
plt.savefig("../Graph2D/1pcCO2/moytemp.eps", format='EPS', dpi=600)
plt.show()
#%%

#%%
print('#######################################################################')
print('Difference temperature moyenne sur 500 ans 2D')
print('#######################################################################')

Tm1pcCO2 = np.zeros((32, 64))
for i in np.arange(0, len(lat)):
    for j in np.arange(0, len(lon)):
        Tm1pcCO2[i, j] = np.mean(Ts1pcCO2[0:500, i, j])

Tdiff2 = Tm1pcCO2 - TmCTRL

map = Basemap(projection='merc', llcrnrlon=-180., llcrnrlat=-80.,
              urcrnrlon=180., urcrnrlat=80., resolution='c')

lons, lats = np.meshgrid(lon-180, lat)

x, y = map(lons, lats)
# Carte de la variable
temp = map.pcolormesh(x, y, Tdiff2, cmap='coolwarm')
map.drawcoastlines() 			# lignes de côte
paralleles = [-60, -30, 0, 30, 60]
meridiens = [-120, -60, 0, 60, 120]
# parallèles et méridiens
map.drawparallels(paralleles, labels=[
    True, False, False, False], fontsize=10)
# labels = [left,right,top,bottom]
map.drawmeridians(meridiens, labels=[
    False, False, False, True], fontsize=10)
cmap = 'coolwarm'
colormesh = map.pcolormesh(x, y, Tdiff2, cmap=cmap)
cb = map.colorbar(colormesh)
#    cb = map.colorbar(temp,"right", size="5%", pad="2%", vmin=220, vmax=310)
cb.set_label('$T$ (K)', fontsize=20)
plt.title('Diff', fontsize=20)
# Sauvegarde
plt.savefig("../Graph2D/Diff/diff500.eps", format='EPS', dpi=600)
plt.show()
#%%

#%%
######################## Questions TP #################################

print(data.variables['tsr'])
print(data.variables['ttr'])

time = data2.variables['time'][:]

TsrCTRL = data.variables['tsr'][:]
TtrCTRL = data.variables['ttr'][:]

Tsr1pcCO2 = data2.variables['tsr'][:]
Ttr1pcCO2 = data2.variables['ttr'][:]

NtCTRL = TsrCTRL - TtrCTRL
Nt1pcCO2 = Ts1pcCO2 - Ttr1pcCO2

NtmCTRL = spatialmean(NtCTRL, lat)
Ntm1pcCO2 = spatialmean(Nt1pcCO2, lat)

ΔN = Ntm1pcCO2 - np.mean(NtmCTRL[900:1000])

plt.plot(t2, ΔN, linewidth=1)
#plt.title('Delta N', fontsize=20)					# Titre
plt.xlabel('t (yr)', fontsize=20)			# Axe des abscisses
plt.ylabel('$\Delta N (W \cdot m^{-2}$)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
# Sauvegarder la figure au format eps
plt.savefig('../Graph1D/DeltaN.eps', format='eps', dpi=600)
plt.show()

#%%

#%%
# Calculer le(s) changement(s) de pente

aa,bb,cc,dd,ee = np.polyfit(Tdiff, ΔN, 4)                          # On construit un polynome de degre 4 pour qu'il suive toutes les données
ΔNfit = aa*Tdiff**4 + bb*Tdiff**3 + cc*Tdiff**2 + dd*Tdiff + ee    # On construit le fit a partir des coefficients du polynome 
plt.scatter(Tdiff, ΔN, linewidth=1)                                # graph des données
plt.plot(Tdiff, ΔNfit, color='r')                                  # graph du fit
plt.show()
dΔN = 4*aa*Tdiff**3 + 3*bb*Tdiff**3 + 2*cc*Tdiff + dd              # On calcule la dérivée du fit pour identifier les changements de pente
plt.scatter(Tdiff, dΔN, color='r')                                 # graph de la dérivée du fit
plt.plot( Tdiff, np.zeros((len(Tdiff))) )                          # On trace une ligne en dΔN = 0
plt.show()
ilist = []
for i in np.arange(0,len(Tdiff)):                                  # On identifie quels éléments de dΔN sont supérieurs à 0 et on met leur indice dans une liste
    o = dΔN[i]
    if o > 0:
        ilist.append(i)
ilist = np.asarray(ilist)
midstart = np.min(ilist)                                           # On définit midstart et midstop comme étant les indices correspondant à l'endroit ou la pente est positive (au milieu du graph 1pcCO2)
midstop = np.max(ilist)
print(midstart, midstop)

#%%

#%%
# λc, Fc, ΔTc

a,b = np.polyfit(Tdiff[0:midstart], ΔN[0:midstart], 1)                             # On construit les fits séparés (des droites) avec les indices correspondant qu'on trouve avec la methode d'au-dessus     
c,d = np.polyfit(Tdiff[midstart:midstop], ΔN[midstart:midstop], 1)
e,f = np.polyfit(Tdiff[midstop:500], ΔN[midstop:500], 1)

ovrslope, ovrintercept = np.polyfit(Tdiff, ΔN, 1)

plt.plot(Tdiff[0:20], a*Tdiff[0:20] + b, linewidth=4, color='r')                   # On trace les données avec les fits
plt.plot(Tdiff[20:100], c*Tdiff[20:100] + d, linewidth=4, color='g')
plt.plot(Tdiff[100:500], e*Tdiff[100:500] + f, linewidth=4, color='y')
plt.plot(Tdiff, ovrslope*Tdiff + ovrintercept, linewidth=4, color='b')
plt.scatter(Tdiff, ΔN, c=time, marker='o')
#plt.title('DeltaN vs DeltaT', fontsize=20)	
plt.xlabel('$\Delta T$ (K)', fontsize=20)				
plt.ylabel('$\Delta N ~(W \cdot m^{-2})$', fontsize=20)				        
#legend = plt.legend(loc='upper center', shadow=True)
plt.viridis()
plt.savefig('../Graph1D/DeltaNvsDeltaT.eps', format='eps', dpi=600)
plt.show()

ΔTc = ovrintercept/(-ovrslope)

print("RED")
print("-slope = λ1 = ", -a)
print("intercept = F1 = ", b)
print("GREEN")
print("-slope = λ2 = ", -c)
print("intercept = F2 = ", d)
print("YELLOW")
print("-slope = λ3 = ", -e)
print("intercept = F3 = ", f)
print("BLUE")
print('-overall slope = λc = ', -ovrslope)
print('overall intecept = Fc = ', ovrintercept)
print('change in temperature at equilibrium (mean over 500 yrs) = ΔTc = ', ΔTc)

#%%

#%%
# λeq, Feq, ΔTeq

Tdiffeq = Tdiff[400:500]
ΔNeq = ΔN[400:500]
time100 = time[400:500]

slopeeq, intercepteq = np.polyfit(Tdiffeq, ΔNeq, 1)

# On trace les données avec les fits

plt.plot(Tdiffeq, slopeeq*Tdiffeq + intercepteq, linewidth=4, color='r')
plt.scatter(Tdiffeq, ΔNeq, c=time100, marker='o')
#plt.title('DeltaN vs DeltaT', fontsize=20)
plt.xlabel('$\Delta T$ (K)', fontsize=20)
plt.ylabel('$\Delta N ~(W \cdot m^{-2})$', fontsize=20)
#legend = plt.legend(loc='upper center', shadow=True)
plt.viridis()
plt.savefig('../Graph1D/DeltaNvsDeltaTeq.eps', format='eps', dpi=600)
plt.show()

ΔTeq = intercepteq/(-slopeeq)

print('λeq = ', -slopeeq)
print('Feq = ', intercepteq)
print('ΔTeq  = ', ΔTeq)


#%%

#%%
# λr, Fr, ΔTr

Tdiffr = Tdiff[0:100]
ΔNr = ΔN[0:100]
time100 = time[0:100]

sloper, interceptr = np.polyfit(Tdiffr, ΔNr, 1)

# On trace les données avec les fits

plt.plot(Tdiffr, sloper*Tdiffr + interceptr, linewidth=4, color='r')
plt.scatter(Tdiffr, ΔNr, c=time100, marker='o')
#plt.title('DeltaN vs DeltaT', fontsize=20)
plt.xlabel('$\Delta T$ (K)', fontsize=20)
plt.ylabel('$\Delta N ~(W \cdot m^{-2})$', fontsize=20)
#legend = plt.legend(loc='upper center', shadow=True)
plt.viridis()
plt.savefig('../Graph1D/DeltaNvsDeltaTr.eps', format='eps', dpi=600)
plt.show()

ΔTr = interceptr/(-sloper)

print('λr = ', -sloper)
print('Fr = ', interceptr)
print('ΔTr  = ', ΔTr)

# %%
