#!/usr/bin/env python
# coding: utf-8

# In[54]:


import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import re


# In[55]:


from docopt import docopt
import numpy as np
import keras
import warnings
#import encode
warnings.filterwarnings("ignore")


# In[56]:


#病毒原文件变为字典 函数
def read_fasta(fasta_file):     
    f = open(fasta_file)
    seq = {}
    for line in f:
        if line.startswith(">"):
            acc = line.strip("\n").strip(">").split(" ")[0]
            seq[acc] = ""
        else:
            seq[acc] += line.strip("\n")
    f.close()

    return seq


# In[57]:


#细菌原文件变为字典 函数
def read_fastab(fasta_file):     
    f = open(fasta_file)
    seq = {}
    for line in f:
        if line.startswith(">"):
            temp = ''.join(re.findall(r'[A-Za-z]', line.strip("\n").strip(">").split(" ")[0])) +line.strip("\n").strip(">").split(" ")[1] + line.strip("\n").strip(">").split(" ")[2]
            if seq.get(temp) is None:
                acc = temp
                seq[acc] = ""
        else:
            seq[acc] += line.strip("\n")
    f.close()

    return seq


# In[58]:


#将原文件中 A、T、C、G用0、1表示

#A、T替换为0，C、G替换为1
def _binary_transfer_AT(seq):
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "1").replace("T", "0")
    seq = ''.join(list(filter(str.isdigit, seq)))  #只保留数字
    
    return seq

#A、C替换为0，T、G替换为1
def _binary_transfer_AC(seq):
    seq = seq.replace("A", "0").replace("C", "0").replace("G", "1").replace("T", "1")
    seq = ''.join(list(filter(str.isdigit, seq)))

    return seq


#将010111二进制编码更新为距离
def _binary_transfer_loc(binary_seq,K):
    loc = []
    for i in range(0, len(binary_seq)-K+1):
        loc.append(int(binary_seq[i:i+K], 2))
    
    return loc


def _loc_transfer_matrix(loc_list, dis, K):
    matrix = np.zeros((2**K, 2**K))
    for i in range(0, len(loc_list)-K-dis):
        matrix[loc_list[i]][loc_list[i+K+dis]] += 1
    
    return matrix

def _matrix_encoding(seq, K):
    seq = seq.upper()
    length = len(seq)
    binary_seq_1 = _binary_transfer_AT(seq)
    binary_seq_2 = _binary_transfer_AC(seq)
    loc_1 = _binary_transfer_loc(binary_seq_1, K)
    loc_2 = _binary_transfer_loc(binary_seq_2, K)
    
    feature = np.hstack((
        _loc_transfer_matrix(loc_1, 0, K).flatten(), _loc_transfer_matrix(loc_2, 0, K).flatten(),
        _loc_transfer_matrix(loc_1, 1, K).flatten(), _loc_transfer_matrix(loc_2, 1, K).flatten(),
        _loc_transfer_matrix(loc_1, 2, K).flatten(), _loc_transfer_matrix(loc_2, 2, K).flatten()))
    
    return feature/(length*1.0) * 100

def matrix_encoding(seq, K):

    return _matrix_encoding(seq, K)


# In[59]:


#特征
def read_fasta2(fasta_file2):
    fasta_file= fasta_file2
    seq_collect = read_fasta(fasta_file)
    genus_feature_list = []
    name = []
    for acc in seq_collect:
        name.append(acc)
        seq = seq_collect[acc]
        genus_feature_list.append(matrix_encoding(seq, 5))
#     x_genus_feature_original = np.array(genus_feature_list).reshape(-1, 32, 64, 6)
    x_genus_feature_original = np.array(genus_feature_list)
    return name,x_genus_feature_original


# In[60]:


#细菌特征
def read_fasta2b(fasta_file2):
    fasta_file= fasta_file2
    seq_collect = read_fastab(fasta_file)
    genus_feature_list = []
    name = []
    for acc in seq_collect:
        name.append(acc)
        seq = seq_collect[acc]
        genus_feature_list.append(matrix_encoding(seq, 5))
#     x_genus_feature_original = np.array(genus_feature_list).reshape(-1, 32, 64, 6)
    x_genus_feature_original = np.array(genus_feature_list)
    return name,x_genus_feature_original


# In[ ]:





# In[61]:


# #使用模型进行预测
# import numpy as np
# from tensorflow.keras.models import load_model

# def predict_result(phage,ac):
#     phagef = read_fasta2(phage)
#     acf = read_fasta2(ac)
    
#     phage_ac = np.hstack((phagef,acf))
#     phage_ac = phage_ac.reshape(-1, 32, 64, 6)
#     model = load_model("/mnt/24t/tz/模型保存/random13389_4834.h5")  #选取自己的.h模型名称
#     a = model.predict(phage_ac)

#     model2 = load_model("/mnt/24t/tz/模型保存/random13389_2034_4834.h5")  #选取自己的.h模型名称
#     b = model2.predict(phage_ac)
#     result = "Possibility of interaction between virus and host," + "Model 1 results：" + str(a[0,1]) + ",Model 2 results：" + str(b[0,1])
    
#     return result


# In[62]:


import numpy as np
from tensorflow.keras.models import load_model

def predict_result(phage,ac):
    model = load_model("./static/upload/modle1.h5")  #选取自己的.h模型名称
    model2 = load_model("./static/upload/model2.h5")  #选取自己的.h模型名称
    
    namep,phagef = read_fasta2(phage)
    namea,acf = read_fasta2b(ac)
    
    result = "Virus,bacteria,Model 1 results,Model 2 results"+ "\r\n"
    for i in range(len(namep)):
        for k in range(len(namea)):
            temp = np.hstack((phagef[i,:],acf[k,:])).reshape(-1, 32, 64, 6)
            a = model.predict(temp)
            b = model2.predict(temp)
            result = result + namep[i] + "," + namea[k] + "," + str(a[0,1]) + "," + str(b[0,1])+ "\r\n"
#            result.append(namep[i] + "-" + namea[k] + "-->" + "Model 1 results：" + str(a[0,1]) + ",Model 2 results：" + str(b[0,1]))
    return result


# In[63]:


#写入文件函数
def write_file(file,seq2,r):
    r_result = ""
    if r<30:
        r_result = "no or few ends"
    elif r<60 and r>30:
        r_result = "with partial ends"
    elif r>60:
        r_result = "end exists."
    with open(file,"w") as f:
        f.write("sequence,frequency,percentage,r-value:," + str(r) + ","+ r_result)
        f.write('\n')
        
        for acc in seq2:
            f.write(acc)
            f.write(",")
            f.write(str(seq2[acc][0]))
            f.write(",")
            f.write(str(seq2[acc][1]))
            f.write('\n')    
    f.close()    
    
#环状病毒
def hzbd(number,genmoe_number,file_s):
    f = open(file_s)
    seq = {}
    line_number = 0
    for line in f:
        line_number = line_number+1        
        if line_number%4 == 2:            
            acc = line[:number]
            if seq.get(acc) is None:
                seq[acc] = []
                seq[acc].append(1)
            elif seq.get(acc) is not None:
                seq[acc][0] = seq[acc][0] + 1
#     print(seq)
    seq2 = dict(sorted(seq.items(), key=lambda x: x[1]))
    avg = (line_number/4)/(2*genmoe_number)
    temp = 0
    for acc in seq2:
        seq2[acc].append(seq2[acc][0]/(line_number/4))
        if temp == 0:
            r = seq2[acc][1]/avg
            temp = 1
    write_file("./static/upload/result_r.csv",seq2,r)        
    write_file("./static/upload/result_r.txt",seq2,r)  
    


# In[64]:


#发邮件函数
import yagmail
import time
import re
def sendemail(email,result_file):
    if len(email) > 7:
        if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) != None:
            yag=yagmail.SMTP(user='ylytzhen@126.com',password='OTPKHWLLMHQYKVRC',host='smtp.126.com')
            yag.send(to=email,subject='Predict the virus host tools that have been running'
                     ,contents='For details, please download the attachment',
            attachments= result_file)


            


# In[65]:


#合并文件
def merge_files(path1,path2):
    path3 = "./static/upload/result_r_result.txt"

    file1 = open(path1,"r",encoding="utf-8",errors="ignore")
    file2 = open(path2,"r",encoding="utf-8",errors="ignore")
    file3 = open(path3,"w",encoding="utf-8",errors="ignore")

    while True:
        mystr1 = file1.readline()#表示一次读取一行
        mystr2 = file2.readline()
        if not mystr1:
        #读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
            break
        #print(mystr,end="")#打印每次读到的内容
        file3.write(mystr1[:-1])
        file3.write("                                            ")
        file3.write(mystr2)

    file1.close()
    file3.close()


# In[ ]:





# In[66]:



UPLOAD_FOLDER = './static/upload'  # 上传到这里
ALLOWED_EXTENSIONS = {'txt','fasta','fna'}  # 允许的格式,保证安全性
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 限制大小64mb


def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':  
        temp = 0
        file = request.files.get("file")        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # filename = request.form['filename']
            # filename = str(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            
        file2 = request.files.get("file2")        
        if file2 and allowed_file(file2.filename):
            filename = secure_filename(file2.filename)
            # filename = request.form['filename']
            # filename = str(filename)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))            
        
        if file and allowed_file(file.filename) and file2 and allowed_file(file2.filename):
            temp = temp +1
            result = predict_result('./static/upload/' + file.filename,'./static/upload/' + file2.filename)
            with open('./static/upload/result.txt','w') as f:            
                f.write(result)            
            with open('./static/upload/result.csv','w') as f:            
                f.write(result)
    #         return result
    #发邮件
            email_address = request.form.get('email')
            result_file = './static/upload/result.csv'
            sendemail(email_address,result_file)

            send_from_directory(app.config['UPLOAD_FOLDER'], 'result.txt')
    
        file3 = request.files.get("file3")        
        if file3 and allowed_file(file3.filename):
            temp = temp+1
            number = int(request.form.get('number'))
            genmoe_number = int(request.form.get('genmoe_number'))

            filename = secure_filename(file3.filename)
            # filename = request.form['filename']
            # filename = str(filename)
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            hzbd(number,genmoe_number,file3.filename)

            email_address = request.form.get('email')
            result_file = "./static/upload/result_r.csv"
            sendemail(email_address,result_file)

        if temp == 1:
            temp = 0
            if file and allowed_file(file.filename) and file2 and allowed_file(file2.filename):
                return redirect(url_for('uploaded_file', filename='result.txt'))
            elif file3 and allowed_file(file3.filename):
                return redirect(url_for('uploaded_file', filename='result_r.txt'))
        elif temp ==2:
            temp = 0
            merge_files("./static/upload/result.txt"
                        ,"./static/upload/result_r.txt")
            return redirect(url_for('uploaded_file', filename='result_r_result.txt'))
        
        
    return render_template("upload.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5005)


# In[ ]:





# In[ ]:




