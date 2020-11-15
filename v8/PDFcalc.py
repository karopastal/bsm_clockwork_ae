######################
#   PDF calculator   #
######################

### Explanations ###
# - Calculate the pdfs by calling an external program

#Import standard
from parton import mkPDF



#Return the PDF of the partons at a given x and q using parton
def PDFxq(x, q, f, pdf):
  return pdf.xfxQ(f, x, q)/x




