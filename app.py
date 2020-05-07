from flask import request, url_for,jsonify,Flask,render_template
import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re
from collections import Counter

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract'

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def model(img):
    text=[]
    l=int(img.shape[0]*1.5)
    w=int(img.shape[1]*1.5)
    img = cv2.resize(img, (w,l))
    ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    #cv2.imwrite('testms2.jpg',dilation)
    
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                    cv2.CHAIN_APPROX_NONE) 
    
    im2 = img.copy() 
    
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        
        cropped = im2[y:y + h, x:x + w] 
        #cv2.imwrite('op.jpg',cropped)

        tet = pytesseract.image_to_string(cropped) 
        text+=[tet.split('\n')]


    temp=[]
    for i in range(len(text)):
        for j in range(len(text[i])):
            temp+=[text[i][j]]
    text=temp

    imp=[]
    fl=0
    for i in range(len(text)):
        text[i]=text[i].split(' ')
        for j in range(len(text[i])):
            text[i][j]=text[i][j].lower()
            if correction(text[i][j])=='mathematics':
                
                imp+=[text[i]]
                fl+=1
                break
            elif correction(text[i][j])=='physics':
                imp+=[text[i]]

                fl+=1
                break
            elif correction(text[i][j])=='chemistry':
                imp+=[text[i]]

                fl+=1
                break
        if fl==3:
            break


    words = ['mathematics','physics','chemistry','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','thirty','fourty','fifty','sixty','seventy','eighty','ninety','hundred']
    final=[]
    for i in range(len(imp)):
        temp=[]
        for j in range(len(imp[i])):
            imp[i][j].replace('|','')
            imp[i][j].replace(']','')
            imp[i][j].replace('[','')
            imp[i][j]=correction(imp[i][j].lower())
            
            if imp[i][j] in words:
                temp+=[imp[i][j]]
        final+=[temp]


    d=dict()
    for i in range(3,23):
        d[words[i]]=i-2
    ct=30
    for i in range(23,len(words)):
        d[words[i]]=ct
        ct+=10


    per=[]
    for i in range(len(final)):
        num=0
        for j in range(len(final[i])):
            if final[i][j] in d:
                num+=d[final[i][j]]
        if num>100:
            num%=100
        per+=[num]        
    result=sum(per)/3


    if result>=90:
        return 'Campus-Alpha'
    elif result>=80 and result<90:
        return 'Campus-Beta'
    elif result>=70 and result<80:
        return 'Campus-Gama'
    else:
        return 'Not Qualified or Image not clear'




##########################
app=Flask('__name__')
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/readDoc", methods=['POST'])

def upload_file():
    if request.method == 'POST':

        #print(request)
        fil = request.files['img'].read()
        npimg = np.fromstring(fil, np.uint8)
        
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        result=model(img)
        #print(img.shape)
        return result
    
if __name__ == "__main__":
    app.run(debug=True,port=33507)
