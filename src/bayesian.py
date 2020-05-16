# Adapted from https://essay.utwente.nl/80505/
from typing import Tuple

import pyAgrum as gum

# InitializetheBayesian Network
from pyAgrum import BayesNet

bn: BayesNet = gum.BayesNet(' MyNetwork')
# Add nodes
R: int = bn.add(gum.LabelizedVariable('R', 'Rain', 2))
S: int = bn.add(gum.LabelizedVariable('S', 'Sprinkler', 2))
G: int = bn.add(gum.LabelizedVariable('G', 'Grass wet', 2))
# Add edges
link: Tuple[int, int]
for link in [(R, S), (R, G), (S, G)]:
    bn.addArc(*link)
# Define conditionalpro ba bi li ty tables
bn.cpt(R)[:] = [0.2, 0.8]
bn.cpt(S)[0, :] = [0.4, 0.6]
bn.cpt(S)[1, :] = [0.01, 0.99]
bn.cpt(G)['R':0, 'S':0] = [0.0, 1.0]
bn.cpt(G)['R':0, 'S':1] = [0.8, 0.2]
bn.cpt(G)['R':1, 'S':0] = [0.9, 0.1]
bn.cpt(G)['R':1, 'S':1] = [0.99, 0.01]
# Createinstanceofthe BNDatabaseGenerator
dbg = gum.BNDatabaseGenerator(bn)
# Sample fromit
dbg.drawSamples(1000)
# Store the samples
dbg.toCSV('File.csv')
