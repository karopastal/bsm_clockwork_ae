##################
#   Background   #
##################

### Explanations ###
# - Interpolate background from a given list

#Import standard
import math
from bisect import bisect_left


#Interpolate background
def BackgroundCal(massBGlist, BGlist, m):

  #Find value to the left
  idx = max(bisect_left(massBGlist, m) - 1, 0)

  #Interpolate
  deltam1 = m - massBGlist[idx]
  deltam2 = massBGlist[idx + 1] - massBGlist[idx]
  inter   = math.log(BGlist[idx])*(1 - deltam1/deltam2) + math.log(BGlist[idx + 1])*deltam1/deltam2

  #Return interpolated value
  return math.exp(inter)




