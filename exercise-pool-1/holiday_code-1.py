import numpy as np ;
import requests ;
import os ;
import sys ;



if __name__ == "__main__":

  ## download MNIST if not present in current dir!
  if os.path.exists("./mnist.npz") == False:
    print ("Downloading MNIST...") ;
    fname = 'mnist.npz'
    url = 'http://www.gepperth.net/alexander/downloads/'
    r = requests.get(url+fname)
    open(fname , 'wb').write(r.content)
  
  ## read it into 'traind' and 'trainl'
  data = np.load("mnist.npz")
  traind = data["arr_0"] ;
  trainl = data["arr_2"] ;
  
  if sys.argv[1] == "1":
    print (traind.shape) ;

  if sys.argv[1] == "2":
    pass ;
    

      



