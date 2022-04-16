import PyPDF2

pdfFileObj = open('sample.pdf', 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

n=pdfReader.numPages

print(n)

textfile= open('textfile.txt','w') 

for i in range(n):
    content=pdfReader.getPage(i).extractText()
    print(content)
    textfile.write(content)
    textfile.write('\n')
    
file=open('textfile.txt','r')

data = file.read()

def searching_Word(sw):
    if sw in data:
        occurrences = data.count(sw)
        return occurrences
    else:
        return "Not Found Matched Word"
        
        
    
  
sw=input("Enter the word to search")
print(searching_Word(sw))







