import PyPDF2

pdfFileObj = open('samp.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages
pageObj = pdfReader.getPage(0)
print pageObj.extractText()
