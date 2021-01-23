#%%

import os

os.environ['PROJ_LIB'] = '/Applications/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj'

# Importation des modules utiles pour les TPs
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from scipy.stats import linregress


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


##### Récupération et observation des variables des fichiers netcdf #####
# Récupération des données du fichier fichier.nc dans la structure data
data = Dataset("/Users/felixlangot/Google Drive (felixlangot@gmail.com)/UVSQ/Modeles_pour_l'etude_des_climats/TP_iLoveClim/felixlangot-CTRL-YR.nc")
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

Tsevol = np.zeros((1000,1))
for i in np.arange(0,1000):
    Tsevol[i] = np.mean(Ts[i,:,:])

Tsevol2 = spatialmean(Ts, lat)

plt.plot(t,Tsevol,linewidth=1)
plt.title('Temperature with np.mean loop', fontsize=20)					# Titre
plt.xlabel('$t$ (yr)', fontsize=20)			# Axe des abscisses
plt.ylabel('$T$ (K)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
#plt.savefig('figure.eps', format='eps', dpi=600)		# Sauvegarder la figure au format eps
plt.show()

plt.plot(t,Tsevol2,linewidth=1)
plt.title('Temperature with spatialmean', fontsize=20)					# Titre
plt.xlabel('$t$ (yr)', fontsize=20)			# Axe des abscisses
plt.ylabel('$T$ (K)', fontsize=20)			        # Axe des ordonnées
#legend = plt.legend(loc='upper center', shadow=True)		# Position et style de légende
#plt.savefig('figure.eps', format='eps', dpi=600)		# Sauvegarder la figure au format eps
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
        plt.savefig("CTRL_Figs/Graph" + str(i) +".png", format='PNG')
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
#    plt.savefig("CTRL_Figs/Graph" + str(i) +".png", format='PNG')
    plt.show()

# %%
